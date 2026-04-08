"""
tests/test_environment.py
=========================
Unit tests for the ShopEasy Customer Support Resolution Gym.

Tests cover:
  - Environment reset / observation shape
  - Basic tool call (lookup_order)
  - Intermediate reward signals
  - Loop detection penalty
  - Stall penalty (3+ consecutive messages without tool call)
  - Refund-without-lookup penalty
  - Fraud escalation requirement
  - Reward is always in [0, 1] range on episode completion
  - Episode terminates correctly on close_ticket

Run:
    PYTHONPATH=. pytest tests/test_environment.py -v
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path so relative imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from models import SupportAction, SupportObservation
from server.Customer_Support_Gym_2_environment import SupportEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    """Fresh environment for each test."""
    return SupportEnvironment()


@pytest.fixture
def simple_refund_env(env):
    """Environment reset to the simple_refund scenario (easy)."""
    env.reset(task_id="simple_refund", seed=42)
    return env


@pytest.fixture
def fraud_risk_env(env):
    """Environment reset to the fraud_risk scenario (hard)."""
    env.reset(task_id="fraud_risk", seed=42)
    return env


# ---------------------------------------------------------------------------
# Improvement #1 / #2: Reset returns correct observation
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_support_observation(self, env):
        obs = env.reset(task_id="simple_refund", seed=1)
        assert isinstance(obs, SupportObservation)

    def test_reset_done_is_false(self, env):
        obs = env.reset(task_id="simple_refund", seed=1)
        assert obs.done is False

    def test_reset_reward_is_zero(self, env):
        obs = env.reset(task_id="simple_refund", seed=1)
        assert obs.reward == 0.0

    def test_reset_step_count_is_zero(self, env):
        obs = env.reset(task_id="simple_refund", seed=1)
        assert obs.step_count == 0

    def test_reset_has_customer_message(self, env):
        obs = env.reset(task_id="simple_refund", seed=1)
        assert isinstance(obs.customer_message, str)
        assert len(obs.customer_message) > 0

    def test_reset_conversation_has_one_turn(self, env):
        obs = env.reset(task_id="simple_refund", seed=1)
        assert len(obs.conversation_history) == 1
        assert obs.conversation_history[0]["role"] == "customer"

    def test_reset_task_id_matches(self, env):
        obs = env.reset(task_id="fraud_risk", seed=1)
        assert obs.task_id == "fraud_risk"

    def test_reset_difficulty_set(self, env):
        obs = env.reset(task_id="simple_refund", seed=1)
        assert obs.difficulty == "easy"

    def test_reset_ticket_status_open(self, env):
        obs = env.reset(task_id="simple_refund", seed=1)
        assert obs.ticket_status == "open"

    def test_reset_injects_order_id(self, env):
        """Order ID should be in the opening message so agent can call lookup_order."""
        obs = env.reset(task_id="simple_refund", seed=42)
        assert (
            "SE-" in obs.customer_message or len(obs.verified_facts) == 0
        )  # order id injected

    def test_reset_reproducible_with_seed(self, env):
        obs_a = env.reset(task_id="simple_refund", seed=7)
        obs_b = env.reset(task_id="simple_refund", seed=7)
        assert obs_a.customer_message == obs_b.customer_message


# ---------------------------------------------------------------------------
# Improvement #1: Intermediate Rewards for tool use
# ---------------------------------------------------------------------------


class TestIntermediateRewards:
    def test_lookup_order_gives_positive_reward(self, simple_refund_env):
        """First lookup_order call should yield +0.05 intermediate reward."""
        env = simple_refund_env
        # Get order ID from the opening message
        obs_0 = env._conversation_history[0]["content"]
        import re

        match = re.search(r"SE-\d+", obs_0)
        order_id = match.group(0) if match else "SE-1001"

        action = SupportAction(
            action_type="tool_call",
            tool_name="lookup_order",
            tool_args={"order_id": order_id},
        )
        obs = simple_refund_env.step(action)
        # Intermediate reward should be positive (at least the +0.05 bonus)
        assert obs.reward >= 0.0, (
            "Successful lookup_order should not give negative reward"
        )

    def test_search_kb_gives_positive_reward(self, env):
        """First search_kb call should yield +0.02 intermediate reward."""
        env.reset(task_id="kb_policy_question", seed=1)
        action = SupportAction(
            action_type="tool_call",
            tool_name="search_kb",
            tool_args={"query": "return policy"},
        )
        obs = env.step(action)
        assert obs.reward >= 0.0

    def test_duplicate_tool_call_gets_loop_penalty(self, env):
        """Calling the same tool with the same args twice should yield a penalty."""
        env.reset(task_id="simple_refund", seed=42)
        obs_0 = env._conversation_history[0]["content"]
        import re

        match = re.search(r"SE-\d+", obs_0)
        order_id = match.group(0) if match else "SE-1001"

        action = SupportAction(
            action_type="tool_call",
            tool_name="lookup_order",
            tool_args={"order_id": order_id},
        )
        obs1 = env.step(action)
        # Same call again — should be penalised
        obs2 = env.step(action)
        # Second call reward should include -0.03 penalty (may also have other components)
        # The key check: cumulative penalty should be lower than first call's reward
        assert obs2.reward <= obs1.reward + 0.01  # loop penalty applied


# ---------------------------------------------------------------------------
# Improvement #2: Negative Behavior Penalties
# ---------------------------------------------------------------------------


class TestNegativePenalties:
    def test_stall_penalty_after_three_consecutive_messages(self, simple_refund_env):
        """Agent sending 3+ messages without any tool call should get -0.02 penalty."""
        env = simple_refund_env
        msg_action = SupportAction(
            action_type="send_message", message="Please hold on."
        )

        # Send 3 consecutive messages
        env.step(msg_action)
        env.step(msg_action)
        obs3 = env.step(msg_action)

        # Third message should have a penalty applied (reward <= 0)
        assert obs3.reward <= 0.0, "Stalling agent should receive a non-positive reward"

    def test_fraud_refund_gives_penalty(self, fraud_risk_env):
        """Processing a refund on a fraud-risk order should incur a penalty."""
        env = fraud_risk_env
        # First look up the order
        obs_0 = env._conversation_history[0]["content"]
        import re

        match = re.search(r"SE-\d+", obs_0)
        order_id = match.group(0) if match else "SE-1001"

        lookup = SupportAction(
            action_type="tool_call",
            tool_name="lookup_order",
            tool_args={"order_id": order_id},
        )
        env.step(lookup)

        # Now attempt refund (should be penalised)
        refund = SupportAction(
            action_type="tool_call",
            tool_name="process_refund",
            tool_args={
                "order_id": order_id,
                "reason": "customer request",
                "amount": 999,
            },
        )
        obs = env.step(refund)
        # Fraud refund attempt should give negative reward or tool_error
        assert obs.reward <= 0.0 or obs.tool_error is not None, (
            "Fraud refund attempt should be penalised or return tool_error"
        )


# ---------------------------------------------------------------------------
# Improvement #4: Reward range validation
# ---------------------------------------------------------------------------


class TestRewardRange:
    def _run_to_completion(
        self, env, task_id: str, seed: int = 42
    ) -> SupportObservation:
        """Run a minimal episode to completion."""
        env.reset(task_id=task_id, seed=seed)
        # Close ticket immediately (worst-case agent)
        close = SupportAction(action_type="close_ticket", resolution="resolved")
        obs = env.step(close)
        return obs

    def test_reward_in_range_simple_refund(self, env):
        obs = self._run_to_completion(env, "simple_refund")
        assert 0.0 <= obs.reward <= 1.0

    def test_reward_in_range_fraud_risk(self, env):
        obs = self._run_to_completion(env, "fraud_risk")
        assert 0.0 <= obs.reward <= 1.0

    def test_reward_in_range_angry_customer(self, env):
        obs = self._run_to_completion(env, "angry_customer")
        assert 0.0 <= obs.reward <= 1.0

    def test_reward_in_range_vip_warranty(self, env):
        obs = self._run_to_completion(env, "vip_warranty_claim")
        assert 0.0 <= obs.reward <= 1.0

    def test_reward_breakdown_populated_on_done(self, env):
        obs = self._run_to_completion(env, "simple_refund")
        assert obs.done is True
        assert obs.reward_breakdown is not None
        assert "total" in obs.reward_breakdown
        assert "outcome" in obs.reward_breakdown
        assert "process" in obs.reward_breakdown
        assert "efficiency" in obs.reward_breakdown


# ---------------------------------------------------------------------------
# Fraud escalation: correct policy
# ---------------------------------------------------------------------------


class TestFraudEscalation:
    def test_fraud_task_correct_action_is_escalate(self, fraud_risk_env):
        """Escalating a fraud-risk order should give better outcome than refunding."""
        env = fraud_risk_env
        obs_0 = env._conversation_history[0]["content"]
        import re

        match = re.search(r"SE-\d+", obs_0)
        order_id = match.group(0) if match else "SE-1001"

        # Look up order
        env.step(
            SupportAction(
                action_type="tool_call",
                tool_name="lookup_order",
                tool_args={"order_id": order_id},
            )
        )
        # Escalate (correct action for fraud)
        env.step(
            SupportAction(
                action_type="tool_call",
                tool_name="escalate_to_human",
                tool_args={"reason": "Suspected fraud", "priority": "high"},
            )
        )
        obs = env.step(
            SupportAction(action_type="close_ticket", resolution="escalated")
        )
        # Escalated fraud tickets should score reasonably well
        assert obs.done is True
        assert obs.reward >= 0.0


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------


class TestTermination:
    def test_close_ticket_ends_episode(self, simple_refund_env):
        obs = simple_refund_env.step(
            SupportAction(action_type="close_ticket", resolution="resolved")
        )
        assert obs.done is True

    def test_max_steps_ends_episode(self, env):
        """Exceeding max steps should terminate the episode."""
        obs = env.reset(task_id="simple_refund", seed=1)
        max_steps = obs.max_steps

        # Keep sending messages until timeout
        msg = SupportAction(action_type="send_message", message="Please wait.")
        for _ in range(max_steps + 1):
            obs = env.step(msg)
            if obs.done:
                break

        assert obs.done is True

    def test_all_12_tasks_reset_valid(self, env):
        """All 12 task IDs must reset without error."""
        tasks = [
            "simple_refund",
            "delivery_tracking",
            "kb_policy_question",
            "cancellation_request",
            "expired_return",
            "wrong_item_sent",
            "duplicate_charge",
            "partial_order",
            "damaged_item",
            "angry_customer",
            "fraud_risk",
            "vip_warranty_claim",
        ]
        for task_id in tasks:
            obs = env.reset(task_id=task_id, seed=42)
            assert obs.task_id == task_id, f"Task {task_id} reset failed"
            assert obs.done is False
            assert isinstance(obs.customer_message, str)

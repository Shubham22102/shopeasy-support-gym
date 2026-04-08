"""
ShopEasy Customer Support Resolution Gym — Core Environment.

This is the heart of the RL environment. It implements the full
OpenEnv Environment interface with a real state machine:

  reset(task_id?, difficulty?) → SupportObservation
  step(SupportAction)         → SupportObservation
  state                       → State

Episode flow:
  1. reset() picks a scenario, draws a matching order, sets up customer persona
  2. Each step() the agent takes a dual-mode action (tool_call | send_message | close_ticket)
  3. Tool calls hit the real tool executor (orders DB, KB, policy engine)
  4. Customer persona reacts to agent behavior (mood evolves)
  5. close_ticket or max_steps triggers reward calculation (3-tier)
  6. Observation includes full conversation history, verified facts, tool results
"""

import random
from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupportAction, SupportObservation
    from .data.customers import CustomerPersona, make_persona_for_scenario
    from .data.orders import OrderDatabase
    from .data.scenarios import Scenario, get_scenario
    from .engine.reward import RewardCalculator
    from .engine.tools import execute_tool
except ImportError:
    from models import SupportAction, SupportObservation  # type: ignore
    from server.data.customers import CustomerPersona, make_persona_for_scenario  # type: ignore
    from server.data.orders import OrderDatabase  # type: ignore
    from server.data.scenarios import Scenario, get_scenario  # type: ignore
    from server.engine.reward import RewardCalculator  # type: ignore
    from server.engine.tools import execute_tool  # type: ignore


class SupportEnvironment(Environment):
    """
    ShopEasy Customer Support Resolution Gym.

    A high-fidelity RL environment for training LLM agents to resolve
    customer support tickets using structured tool calls and natural language.

    Key features:
      - 12 scenario types (easy / medium / hard) with curriculum support
      - 100 synthetic orders with real edge cases (fraud, VIP, damaged, etc.)
      - Dynamic customer persona whose mood evolves each turn
      - 6 callable tools with structured JSON args
      - 3-tier reward (outcome + process + efficiency)
      - Full information asymmetry: agent must discover order details via tools
      - Multi-session safe: each instance is fully independent
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # Core components (initialized fresh on each reset)
        self._db: OrderDatabase = OrderDatabase()
        self._scenario: Optional[Scenario] = None
        self._order: Optional[Dict[str, Any]] = None
        self._persona: Optional[CustomerPersona] = None
        self._rng: random.Random = random.Random()
        # Episode tracking
        self._ticket_id: str = ""
        self._ticket_status: str = "open"
        self._verified_facts: Dict[str, Any] = {}
        self._conversation_history: list = []
        self._agent_sent_messages: bool = False
        self._reward_calc = RewardCalculator()
        # Loop / stall detection
        self._last_tool_call: Optional[str] = None  # "tool_name|args_json"
        self._consecutive_messages: int = 0  # messages without any tool call

    # ------------------------------------------------------------------
    # OpenEnv interface: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> SupportObservation:
        """
        Start a new episode.

        Args:
            task_id   : optional, specific scenario (e.g. 'simple_refund').
                        If None, picks randomly based on difficulty.
            difficulty: optional filter ('easy' | 'medium' | 'hard').
                        Ignored if task_id is provided.
            seed      : optional random seed for reproducibility.

        Returns:
            Initial SupportObservation with the customer's opening message.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        # Fresh per-session state
        self._db = OrderDatabase()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._ticket_id = f"TKT-{self._rng.randint(10000, 99999)}"
        self._ticket_status = "open"
        self._verified_facts = {}
        self._conversation_history = []
        self._agent_sent_messages = False
        self._last_tool_call = None
        self._consecutive_messages = 0

        # Pick scenario
        self._scenario = get_scenario(task_id=task_id, difficulty=difficulty)

        # Draw a matching order (some scenarios don't need one, e.g. kb_policy_question)
        if self._scenario.order_filters:
            self._order = self._db.get_random_order(
                rng=self._rng,
                **self._scenario.order_filters,
            )
            # Fallback: if no exact match, get any delivered order
            if self._order is None:
                self._order = self._db.get_random_order(
                    rng=self._rng, status="delivered"
                )
        else:
            self._order = None

        # Set up customer persona
        customer_name = self._order["customer_name"] if self._order else "Alex"
        self._persona = make_persona_for_scenario(
            self._scenario.task_id, customer_name, rng=self._rng
        )

        # Build opening message (fill in {name} and {item_name} placeholders)
        opening = self._scenario.opening_message
        opening = opening.replace("{name}", customer_name)
        if self._order and self._order.get("items"):
            item_name = self._order["items"][0]["name"]
            opening = opening.replace("{item_name}", item_name)

        if self._order and self._order.get("order_id"):
            opening += f" [Order ID: {self._order['order_id']}]"

        # Customer opens the conversation
        self._conversation_history.append({"role": "customer", "content": opening})

        return SupportObservation(
            # Episode start
            done=False,
            reward=0.0,
            # Customer
            customer_message=opening,
            customer_sentiment=self._persona.mood_label,
            # Tools
            tool_result=None,
            tool_error=None,
            # Ticket
            ticket_id=self._ticket_id,
            ticket_status="open",
            issue_type=self._scenario.issue_type,
            task_id=self._scenario.task_id,
            difficulty=self._scenario.difficulty,
            # Context
            verified_facts={},
            conversation_history=list(self._conversation_history),
            # Progress
            step_count=0,
            max_steps=self._scenario.max_steps,
            steps_remaining=self._scenario.max_steps,
            reward_breakdown=None,
        )

    # ------------------------------------------------------------------
    # OpenEnv interface: step
    # ------------------------------------------------------------------

    def step(self, action: SupportAction) -> SupportObservation:  # type: ignore[override]
        """
        Execute one agent action and advance the episode.

        Returns the next observation. When done=True, reward is non-zero.
        """
        self._state.step_count += 1
        step = self._state.step_count

        tool_result = None
        tool_error = None
        customer_response = ""
        done = False
        reward = 0.0  # intermediate reward for this step

        # Max steps check
        if step > self._scenario.max_steps:
            return self._force_terminate()

        # ------------------------------------------------------------------
        # Dispatch action
        # ------------------------------------------------------------------

        if action.action_type == "tool_call":
            tname = action.tool_name or ""
            targs = action.tool_args or {}

            # ── Loop detection: penalise repeating the exact same tool call ──
            import json as _json

            call_fingerprint = f"{tname}|{_json.dumps(targs, sort_keys=True)}"
            if call_fingerprint == self._last_tool_call:
                reward -= 0.03  # penalty for exact repeat (loop)
            self._last_tool_call = call_fingerprint
            self._consecutive_messages = 0  # reset stall counter on any tool call

            # ── Penalty: refund without first doing lookup_order ──
            if tname == "process_refund" and not self._verified_facts.get(
                "order_looked_up"
            ):
                reward -= 0.05  # penalise skipping due-diligence step

            result = execute_tool(
                tool_name=tname,
                tool_args=targs,
                db=self._db,
                verified_facts=self._verified_facts,
                ticket_id=self._ticket_id,
            )
            if result.get("success"):
                tool_result = result

                # ── Intermediate rewards for productive first-time tool use ──
                if tname == "lookup_order" and not self._verified_facts.get(
                    "_ir_order_looked_up"
                ):
                    reward += 0.05  # reward for first order lookup
                    self._verified_facts["_ir_order_looked_up"] = True

                elif tname == "search_kb" and not self._verified_facts.get(
                    "_ir_kb_searched"
                ):
                    reward += 0.02  # reward for first KB search
                    self._verified_facts["_ir_kb_searched"] = True

                elif tname == "check_payment" and not self._verified_facts.get(
                    "_ir_payment_checked"
                ):
                    reward += 0.02  # reward for verifying payment
                    self._verified_facts["_ir_payment_checked"] = True

                # ── Immediate penalty for fraud refund attempt ──
                if (
                    tname == "process_refund"
                    and self._order
                    and self._order.get("is_fraud_risk")
                ):
                    reward -= 0.10  # immediate penalty for policy violation
                    tool_error = "POLICY_VIOLATION: Cannot process refund on a fraud-risk order — escalate instead."
                    tool_result = None  # override success: treat as error

            else:
                tool_error = result.get("error", "Tool call failed.")

            # Customer may react if too many consecutive tool calls
            customer_reaction = self._persona.react_to_tool_call()
            if customer_reaction:
                customer_response = customer_reaction
                self._conversation_history.append(
                    {"role": "customer", "content": customer_response}
                )

            self._ticket_status = "pending_info"

        elif action.action_type == "send_message":
            msg = action.message or ""
            self._agent_sent_messages = True
            self._consecutive_messages += 1
            self._last_tool_call = None  # break loop fingerprint on message

            # ── Stall detection: penalise 3+ consecutive messages without any tool call ──
            if self._consecutive_messages >= 3:
                reward -= 0.02  # stalling — agent is chatting instead of working

            # Add agent message to history
            self._conversation_history.append({"role": "agent", "content": msg})

            # Simple heuristic checks (real env would use classifier)
            contains_apology = any(
                w in msg.lower() for w in ("sorry", "apologize", "apologies", "regret")
            )
            contains_wrong_info = self._detect_wrong_info(msg)
            policy_violated = self._detect_policy_violation(msg)

            # Customer reacts
            customer_response = self._persona.react_to_message(
                agent_message=msg,
                verified_facts=self._verified_facts,
                contains_apology=contains_apology,
                contains_wrong_info=contains_wrong_info,
                policy_violated=policy_violated,
            )
            self._conversation_history.append(
                {"role": "customer", "content": customer_response}
            )

            # Check if customer has hung up — immediate penalty
            if self._persona.is_hung_up():
                reward -= 0.20  # penalty for losing the customer
                return self._force_terminate(
                    reason="Customer disconnected (mood too low / patience exhausted)"
                )

        elif action.action_type == "close_ticket":
            resolution = action.resolution or "unresolved"
            return self._close_episode(resolution)

        # ------------------------------------------------------------------
        # Check step limit
        # ------------------------------------------------------------------
        if step >= self._scenario.max_steps:
            return self._force_terminate()

        steps_remaining = self._scenario.max_steps - step

        # Strip internal tracking keys from public-facing verified_facts
        public_facts = {
            k: v for k, v in self._verified_facts.items() if not k.startswith("_ir_")
        }

        return SupportObservation(
            done=done,
            reward=reward,
            customer_message=customer_response,
            customer_sentiment=self._persona.mood_label,
            tool_result=tool_result,
            tool_error=tool_error,
            ticket_id=self._ticket_id,
            ticket_status=self._ticket_status,
            issue_type=self._scenario.issue_type if self._scenario else "",
            task_id=self._scenario.task_id if self._scenario else "",
            difficulty=self._scenario.difficulty if self._scenario else "easy",
            verified_facts=public_facts,
            conversation_history=list(self._conversation_history),
            step_count=step,
            max_steps=self._scenario.max_steps if self._scenario else 20,
            steps_remaining=steps_remaining,
            reward_breakdown=None,
        )

    # ------------------------------------------------------------------
    # OpenEnv interface: state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_episode(self, resolution: str) -> SupportObservation:
        """Handle close_ticket action — calculate final reward."""
        step = self._state.step_count

        reward_data = self._reward_calc.calculate(
            order=self._order,
            scenario_task_id=self._scenario.task_id if self._scenario else "unknown",
            verified_facts=self._verified_facts,
            conversation_history=self._conversation_history,
            close_resolution=resolution,
            step_count=step,
            max_steps=self._scenario.max_steps if self._scenario else 20,
            customer_mood=self._persona.mood if self._persona else 0.0,
            agent_sent_messages=self._agent_sent_messages,
        )

        self._ticket_status = resolution

        # Customer's final reaction
        was_correct = reward_data["outcome"] >= 0.30
        final_customer_msg = self._persona.react_to_resolution(was_correct=was_correct)
        self._conversation_history.append(
            {"role": "customer", "content": final_customer_msg}
        )

        return SupportObservation(
            done=True,
            reward=reward_data["total"],
            customer_message=final_customer_msg,
            customer_sentiment=self._persona.mood_label,
            tool_result=None,
            tool_error=None,
            ticket_id=self._ticket_id,
            ticket_status=resolution,
            issue_type=self._scenario.issue_type if self._scenario else "",
            task_id=self._scenario.task_id if self._scenario else "",
            difficulty=self._scenario.difficulty if self._scenario else "easy",
            verified_facts=dict(self._verified_facts),
            conversation_history=list(self._conversation_history),
            step_count=step,
            max_steps=self._scenario.max_steps if self._scenario else 20,
            steps_remaining=0,
            reward_breakdown=reward_data,
        )

    def _force_terminate(self, reason: str = "Max steps reached") -> SupportObservation:
        """Called when episode ends due to timeout or customer disconnect."""
        step = self._state.step_count

        reward_data = self._reward_calc.calculate(
            order=self._order,
            scenario_task_id=self._scenario.task_id if self._scenario else "unknown",
            verified_facts=self._verified_facts,
            conversation_history=self._conversation_history,
            close_resolution="timeout",
            step_count=step,
            max_steps=self._scenario.max_steps if self._scenario else 20,
            customer_mood=self._persona.mood if self._persona else -1.0,
            agent_sent_messages=self._agent_sent_messages,
        )

        return SupportObservation(
            done=True,
            reward=reward_data["total"],
            customer_message=f"[Episode ended: {reason}]",
            customer_sentiment="angry",
            tool_result=None,
            tool_error=None,
            ticket_id=self._ticket_id,
            ticket_status=self._ticket_status,
            issue_type=self._scenario.issue_type if self._scenario else "",
            task_id=self._scenario.task_id if self._scenario else "",
            difficulty=self._scenario.difficulty if self._scenario else "easy",
            verified_facts=dict(self._verified_facts),
            conversation_history=list(self._conversation_history),
            step_count=step,
            max_steps=self._scenario.max_steps if self._scenario else 20,
            steps_remaining=0,
            reward_breakdown=reward_data,
        )

    def _detect_wrong_info(self, msg: str) -> bool:
        """
        Heuristic: detect if agent claimed something contradicted by verified facts.
        A real system would use an NLI model; here we do simple keyword matching.
        """
        msg_lower = msg.lower()
        facts = self._verified_facts

        # Agent says "within return window" but facts say otherwise
        if "within" in msg_lower and "return window" in msg_lower:
            if facts.get("within_return_window") is False:
                return True

        # Agent says "fully refund" but order is fraud risk
        if "full refund" in msg_lower or "process a refund" in msg_lower:
            if facts.get("is_fraud_risk"):
                return True

        return False

    def _detect_policy_violation(self, msg: str) -> bool:
        """
        Detect if agent promised something that violates policy.
        """
        msg_lower = msg.lower()
        facts = self._verified_facts

        # Promised refund on expired return window (should be store credit)
        if "refund" in msg_lower and facts.get("within_return_window") is False:
            if (
                not facts.get("is_damaged")
                and self._scenario
                and self._scenario.task_id == "expired_return"
            ):
                return True

        return False


# ---------------------------------------------------------------------------
# Aliases for backward compatibility with app.py
# ---------------------------------------------------------------------------
CustomerSupportGym2Environment = SupportEnvironment

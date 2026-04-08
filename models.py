"""
Data models for the ShopEasy Customer Support Resolution Gym.

Defines the dual-mode SupportAction and rich SupportObservation that
mirror real-world LLM tool-use APIs (structured JSON actions with args).
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator
import json


# ---------------------------------------------------------------------------
# ACTION  — what the agent can do each step
# ---------------------------------------------------------------------------


class SupportAction(Action):
    """
    Dual-mode action for the ShopEasy Customer Support environment.

    The agent must choose ONE of three action types per step:
      - tool_call    : invoke a backend tool (lookup, refund, search, escalate)
      - send_message : send a natural-language reply to the customer
      - close_ticket : formally close the support ticket with a resolution code
    """

    action_type: Literal["tool_call", "send_message", "close_ticket"] = Field(
        ...,
        description=(
            "Type of action. "
            "'tool_call' to call a backend tool; "
            "'send_message' to reply to the customer; "
            "'close_ticket' to end the episode."
        ),
    )

    # --- tool_call fields ---
    tool_name: Optional[
        Literal[
            "lookup_order",
            "process_refund",
            "search_kb",
            "escalate_to_human",
            "cancel_subscription",
            "check_payment",
        ]
    ] = Field(
        None,
        description=(
            "Name of the tool to call. Required when action_type='tool_call'. "
            "lookup_order: retrieve order details by order_id. "
            "process_refund: initiate a refund for an order. "
            "search_kb: search knowledge base articles by keyword query. "
            "escalate_to_human: escalate ticket to a human supervisor. "
            "cancel_subscription: cancel a customer subscription. "
            "check_payment: verify payment details for duplicate/failed charges."
        ),
    )
    tool_args: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Arguments for the tool call as a dict. "
            "For lookup_order: {'order_id': 'SE-XXXX'}. "
            "For process_refund: {'order_id': 'SE-XXXX', 'reason': '...', 'amount': float}. "
            "For search_kb: {'query': 'refund policy electronics'}. "
            "For escalate_to_human: {'reason': '...', 'priority': 'normal'|'high'}. "
            "For cancel_subscription: {'subscription_id': 'SUB-XXXX', 'reason': '...'}. "
            "For check_payment: {'order_id': 'SE-XXXX'}."
        ),
    )

    # --- send_message fields ---
    message: Optional[str] = Field(
        None,
        description="Natural-language message to send to the customer. Required when action_type='send_message'.",
    )

    # --- close_ticket fields ---
    resolution: Optional[Literal["resolved", "escalated", "unresolved"]] = Field(
        None,
        description="Resolution code for closing the ticket. Required when action_type='close_ticket'.",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_tool_args_string(cls, data: Any) -> Any:
        if isinstance(data, dict):
            tool_args = data.get("tool_args")
            if isinstance(tool_args, str):
                try:
                    data["tool_args"] = json.loads(tool_args)
                except json.JSONDecodeError:
                    pass  # Let Pydantic handle the error natively if it's invalid JSON
        return data


# ---------------------------------------------------------------------------
# OBSERVATION  — what the agent sees after each step
# ---------------------------------------------------------------------------


class SupportObservation(Observation):
    """
    Rich observation from the ShopEasy Customer Support environment.

    Includes the customer's latest message, emotional state, tool results,
    ticket status, and full conversation history — everything the agent
    needs to make a good next decision.
    """

    # What the customer just said
    customer_message: str = Field(
        default="",
        description="The customer's latest message or opening complaint.",
    )
    customer_sentiment: Literal["calm", "frustrated", "angry", "satisfied"] = Field(
        default="calm",
        description="Inferred emotional state of the customer this turn.",
    )

    # Result of last tool call (populated only if action was tool_call)
    tool_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured result returned by the last tool call, or None.",
    )
    tool_error: Optional[str] = Field(
        default=None,
        description="Error message if the last tool call failed, or None.",
    )

    # Ticket / episode state
    ticket_id: str = Field(
        default="",
        description="Unique identifier for this support ticket.",
    )
    ticket_status: Literal[
        "open", "pending_info", "pending_refund", "resolved", "escalated"
    ] = Field(
        default="open",
        description="Current lifecycle status of the support ticket.",
    )
    issue_type: str = Field(
        default="",
        description=(
            "Category of the customer's issue, e.g. "
            "'refund_request', 'delivery_issue', 'wrong_item', 'duplicate_charge', "
            "'subscription_cancel', 'warranty_claim', 'fraud_report'."
        ),
    )
    task_id: str = Field(
        default="",
        description="Identifier of the scenario/task being run (e.g. 'simple_refund').",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Difficulty level of this episode.",
    )

    # What the agent has confirmed via tools
    verified_facts: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Key facts the agent has confirmed by calling tools. "
            "E.g. {'order_status': 'delivered', 'within_return_window': True}."
        ),
    )

    # Full conversation so far
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Ordered list of turns: [{'role': 'customer'|'agent', 'content': str}]. "
            "Grows each step."
        ),
    )

    # Episode progress
    step_count: int = Field(default=0, description="Number of steps taken so far.")
    max_steps: int = Field(
        default=20, description="Maximum steps allowed for this episode."
    )
    steps_remaining: int = Field(
        default=20, description="Steps remaining before forced termination."
    )

    # Reward breakdown (populated on close_ticket or forced termination)
    reward_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Detailed reward breakdown: "
            "{'outcome': float, 'process': float, 'efficiency': float, 'total': float}. "
            "Populated only at episode end."
        ),
    )


# ---------------------------------------------------------------------------
# Keep legacy aliases so app.py doesn't break until we update it
# ---------------------------------------------------------------------------
CustomerSupportGym2Action = SupportAction
CustomerSupportGym2Observation = SupportObservation

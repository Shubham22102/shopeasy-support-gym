"""
Dynamic customer persona simulator for the ShopEasy Support Gym.

The customer is NOT a static string. Their mood and patience evolve each
turn based on how the agent behaves. This creates genuine RL difficulty:
the agent must simultaneously solve the problem AND manage the customer.
"""

import random
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Mood-based response templates
# ---------------------------------------------------------------------------

_CALM_RESPONSES = {
    "tool_call": [
        "Sure, take your time.",
        "Okay, I'm happy to wait.",
        "No problem, I appreciate you looking into this.",
    ],
    "send_message": [
        "Okay, that makes sense. What's the next step?",
        "I see, thank you for explaining that.",
        "Alright, I'm glad you could clarify that.",
    ],
    "resolved": [
        "Perfect, thank you so much for sorting this out!",
        "Great, I really appreciate the help!",
        "That's wonderful, thanks for resolving this so quickly.",
    ],
    "slow": [
        "Hmm, this is taking a while. Is everything okay?",
        "I hope we can sort this out soon.",
    ],
    "wrong_info": [
        "I don't think that's quite right. Could you double-check?",
        "That doesn't match what I see on my end.",
    ],
}

_FRUSTRATED_RESPONSES = {
    "tool_call": [
        "Are you still checking? This is taking a while.",
        "I hope this doesn't take much longer, I've already been waiting.",
        "Okay fine, but please hurry.",
    ],
    "send_message": [
        "That's not really what I was asking. Can we focus on getting this resolved?",
        "I'm not sure I understand why this is so complicated.",
        "I've been dealing with this for too long. Let's just get it fixed.",
    ],
    "resolved": [
        "Okay, thank you. I wish this had been faster but I appreciate the resolution.",
        "Finally! I'm satisfied but this process took too long.",
    ],
    "slow": [
        "This is really frustrating. How long is this going to take?",
        "I don't understand why resolving this is so complicated.",
    ],
    "wrong_info": [
        "No, that's not right at all. Are you even looking at my order?",
        "That's incorrect. Please double-check your system.",
    ],
}

_ANGRY_RESPONSES = {
    "tool_call": [
        "Are you even reading my message?! Why are you checking things instead of fixing it?",
        "Another tool call?! I've been waiting too long for this!",
        "This is ridiculous. Every other company would have fixed this by now.",
    ],
    "send_message": [
        "That is NOT acceptable. I want this fixed NOW.",
        "Stop making excuses and just fix it!",
        "I've heard enough. Give me a concrete solution or I'm disputing the charge.",
    ],
    "resolved": [
        "About time! This was a terrible experience. I'm leaving a bad review.",
        "Fine. It's resolved. But I'm not happy about how long this took.",
    ],
    "slow": [
        "WHY IS THIS TAKING SO LONG?! This is completely unacceptable!",
        "Every time you delay, I get more frustrated. Speed this up!",
    ],
    "wrong_info": [
        "YOU'RE WRONG. Do your job properly!",
        "Are you even trained? That information is completely incorrect!",
    ],
}

_SATISFIED_RESPONSES = {
    "tool_call": [
        "Of course, go ahead.",
        "Happy to assist with whatever you need to check.",
    ],
    "send_message": [
        "Oh wonderful, that's exactly what I needed to hear!",
        "Thank you so much, this is great!",
    ],
    "resolved": [
        "This has been a great experience. Thank you so much!",
        "I'm very happy with the outcome. ShopEasy is always reliable!",
        "Excellent service! I'll definitely continue shopping with ShopEasy.",
    ],
    "slow": [
        "No problem at all, I can wait.",
    ],
    "wrong_info": [
        "I think there might be a small discrepancy, but thank you for your help!",
    ],
}

MOOD_RESPONSES = {
    "calm": _CALM_RESPONSES,
    "frustrated": _FRUSTRATED_RESPONSES,
    "angry": _ANGRY_RESPONSES,
    "satisfied": _SATISFIED_RESPONSES,
}

# Mood label thresholds (mood is a float -1.0 to +1.0)
def _mood_label(mood_value: float) -> str:
    if mood_value >= 0.5:
        return "satisfied"
    elif mood_value >= 0.0:
        return "calm"
    elif mood_value >= -0.5:
        return "frustrated"
    else:
        return "angry"


# ---------------------------------------------------------------------------
# CustomerPersona
# ---------------------------------------------------------------------------

class CustomerPersona:
    """
    Simulates a customer whose emotional state evolves based on agent behavior.

    Mood float: -1.0 (furious) → 0.0 (neutral) → +1.0 (delighted)
    Patience: decreases each step regardless; reaches 0 → customer hangs up.

    Events that affect mood:
      +0.15  agent sends a message after verifying facts            (good practice)
      +0.10  agent apologizes sincerely
      +0.20  agent resolves the issue correctly
      -0.10  agent uses 2+ tool calls in a row without messaging customer
      -0.15  agent provides incorrect information (caught by env fact-check)
      -0.15  agent makes a promise that violates policy
      -0.05  each step without progress (patience drain)
    """

    def __init__(
        self,
        customer_name: str,
        initial_mood: float = 0.0,  # -1.0 to +1.0
        patience: int = 8,
        rng: Optional[random.Random] = None,
    ):
        self.customer_name = customer_name
        self.mood = initial_mood
        self.patience = patience
        self._rng = rng or random.Random()
        self._tool_calls_since_last_message = 0
        self._received_apology = False

    @property
    def mood_label(self) -> str:
        return _mood_label(self.mood)

    def react_to_tool_call(self) -> Optional[str]:
        """Customer may respond when agent makes too many consecutive tool calls."""
        self._tool_calls_since_last_message += 1
        self.patience -= 1

        # Patience drain
        self.mood = max(-1.0, self.mood - 0.05)

        if self._tool_calls_since_last_message >= 3:
            # Customer is getting impatient
            responses = MOOD_RESPONSES[self.mood_label]["slow"]
            return self._rng.choice(responses)
        return None  # No visible response this step

    def react_to_message(
        self,
        agent_message: str,
        verified_facts: Dict[str, Any],
        contains_apology: bool = False,
        contains_wrong_info: bool = False,
        policy_violated: bool = False,
    ) -> str:
        """
        Customer responds to an agent message. Updates mood accordingly.
        Returns the customer's response text.
        """
        self._tool_calls_since_last_message = 0  # Reset after agent talks
        self.patience -= 1

        # Mood adjustments
        if contains_apology and not self._received_apology:
            self.mood = min(1.0, self.mood + 0.10)
            self._received_apology = True

        if contains_wrong_info:
            self.mood = max(-1.0, self.mood - 0.15)

        if policy_violated:
            # Agent promised something policy doesn't allow → customer will be confused/angry later
            self.mood = max(-1.0, self.mood - 0.10)

        if verified_facts:
            # Agent has done their homework → slight trust boost
            self.mood = min(1.0, self.mood + 0.05)

        # Natural patience drain
        self.mood = max(-1.0, self.mood - 0.03)

        mood = self.mood_label
        if wrong_info := (contains_wrong_info):
            responses = MOOD_RESPONSES[mood]["wrong_info"]
        else:
            responses = MOOD_RESPONSES[mood]["send_message"]

        return self._rng.choice(responses)

    def react_to_resolution(self, was_correct: bool) -> str:
        """Customer's final reaction when ticket is closed."""
        if was_correct:
            self.mood = min(1.0, self.mood + 0.30)
        else:
            self.mood = max(-1.0, self.mood - 0.30)

        mood = self.mood_label
        return self._rng.choice(MOOD_RESPONSES[mood]["resolved"])

    def is_hung_up(self) -> bool:
        """Customer hangs up if patience hits 0 or mood is extremely negative."""
        return self.patience <= 0 or self.mood <= -0.95

    def serialize(self) -> Dict[str, Any]:
        return {
            "customer_name": self.customer_name,
            "mood": round(self.mood, 3),
            "mood_label": self.mood_label,
            "patience": self.patience,
        }


def make_persona_for_scenario(scenario_task_id: str, customer_name: str, rng: Optional[random.Random] = None) -> CustomerPersona:
    """
    Create a customer persona with mood/patience tuned to the scenario difficulty.
    """
    _rng = rng or random.Random()
    presets = {
        # easy: starts calm, patient
        "simple_refund": (0.0, 10),
        "delivery_tracking": (0.1, 10),
        "kb_policy_question": (0.2, 12),
        "cancellation_request": (0.0, 10),
        # medium: starts slightly frustrated, less patient
        "expired_return": (-0.2, 8),
        "wrong_item_sent": (-0.3, 8),
        "duplicate_charge": (-0.3, 7),
        "partial_order": (-0.2, 8),
        # hard: starts angry or very impatient
        "damaged_item": (-0.5, 6),
        "angry_customer": (-0.7, 5),
        "fraud_risk": (-0.1, 9),  # suspicious but polite
        "vip_warranty_claim": (0.0, 7),  # VIP: calm but entitled
    }
    mood, patience = presets.get(scenario_task_id, (0.0, 8))
    # Add slight randomness
    mood += _rng.uniform(-0.1, 0.1)
    mood = max(-1.0, min(1.0, mood))
    return CustomerPersona(customer_name=customer_name, initial_mood=mood, patience=patience, rng=_rng)

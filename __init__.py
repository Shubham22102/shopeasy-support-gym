"""ShopEasy Customer Support Resolution Gym — OpenEnv Environment."""

from .client import CustomerSupportGym2Env
from .models import (
    CustomerSupportGym2Action,
    CustomerSupportGym2Observation,
    SupportAction,
    SupportObservation,
)

__all__ = [
    # Primary names
    "SupportAction",
    "SupportObservation",
    "CustomerSupportGym2Env",
    # Legacy aliases
    "CustomerSupportGym2Action",
    "CustomerSupportGym2Observation",
]

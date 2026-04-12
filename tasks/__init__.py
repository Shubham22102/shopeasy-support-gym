"""Task registry and grader exports for OpenEnv validation."""

# Import your grading logic
from reward.grader import grade, get_task_graders, TaskGrader

# THE FIX: Explicitly define the TASKS list here so the Python validator sees them
TASKS = [
    {
        "id": "simple_refund",
        "name": "Simple Refund",
        "difficulty": "easy"
    },
    {
        "id": "delivery_tracking",
        "name": "Delivery Tracking",
        "difficulty": "easy"
    },
    {
        "id": "fraud_risk",
        "name": "Fraud Risk Handling",
        "difficulty": "hard"
    }
]

__all__ = [
    "TASKS",
    "grade",
    "get_task_graders",
    "TaskGrader"
]
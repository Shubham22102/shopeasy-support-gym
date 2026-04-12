"""Task registry and grader exports for OpenEnv validation."""

# Import the fully clamped grading logic from your reward engine
from reward.grader import grade, get_task_graders, TaskGrader

# Explicitly define the baseline tasks here so the Python validator sees them.
# The validator requires at least 3 to pass the pipeline.
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

# Export everything exactly as OpenEnv expects
__all__ = [
    "TASKS",
    "grade",
    "get_task_graders",
    "TaskGrader"
]
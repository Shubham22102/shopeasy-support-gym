"""Task registry and grader exports for OpenEnv validation."""

# Import the grading logic from your reward/grader.py file
from reward.grader import grade, get_task_graders, TaskGrader

# Explicitly export them so OpenEnv's auto-discovery can find them
__all__ = [
    "grade",
    "get_task_graders",
    "TaskGrader"
]
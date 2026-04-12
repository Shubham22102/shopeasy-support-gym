# In tasks/__init__.py

# Point this to wherever grader.py actually lives. 
# If the folder is named 'reward' (singular), use 'from reward.grader import ...'
from reward.grader import grade, get_task_graders, TaskGrader

__all__ = [
    "grade",
    "get_task_graders",
    "TaskGrader"
]
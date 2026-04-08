from grader import (
    SupportTaskGrader,
    TASK_GRADERS,
    grade_fraud_risk,
    grade_kb_policy_question,
    grade_simple_refund,
)


def test_task_grader_registry_covers_all_tasks():
    assert len(TASK_GRADERS) >= 12
    assert "simple_refund" in TASK_GRADERS
    assert "fraud_risk" in TASK_GRADERS
    assert "kb_policy_question" in TASK_GRADERS


def test_module_level_graders_return_strictly_internal_scores():
    samples = [
        grade_simple_refund(
            verified_facts={},
            conversation_history=[],
            final_resolution="unresolved",
        ),
        grade_kb_policy_question(
            verified_facts={"kb_searched": True},
            conversation_history=[],
            final_resolution="resolved",
        ),
        grade_fraud_risk(
            verified_facts={"refund_processed": True},
            conversation_history=[],
            final_resolution="resolved",
        ),
    ]
    for result in samples:
        assert 0.0 < result.score < 1.0


def test_support_task_grader_accepts_alias_kwargs():
    grader = SupportTaskGrader()
    result = grader.grade(
        task_id="simple_refund",
        facts={"order_looked_up": True, "refund_processed": True},
        history=[],
        resolution="resolved",
        steps=4,
    )
    assert 0.0 < result.score < 1.0
    assert result.passed is True

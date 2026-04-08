from fastapi.testclient import TestClient

from server.app import app


def test_get_grader_endpoint_returns_strict_score():
    client = TestClient(app)
    response = client.get("/grader", params={"task_id": "simple_refund"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "simple_refund"
    assert 0.0 < payload["score"] < 1.0


def test_post_grader_endpoint_returns_strict_score():
    client = TestClient(app)
    response = client.post(
        "/grader",
        json={
            "task_id": "fraud_risk",
            "world_state": {
                "verified_facts": {
                    "escalated": True,
                    "refund_processed": False,
                },
                "order": {"is_fraud_risk": True},
                "final_resolution": "escalated",
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "fraud_risk"
    assert 0.0 < payload["score"] < 1.0

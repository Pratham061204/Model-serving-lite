import os

os.environ["TEST_MODE"] = "1"

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_prediction():
    response = client.post(
        "/predict",
        json={"text": "I love this app, it is amazing"}
    )

    data = response.json()

    assert response.status_code == 200
    assert "request_id" in data
    assert data["label"] == "POSITIVE"
    assert "score" in data
    assert "model_latency_ms" in data
    assert "total_latency_ms" in data
    assert "drift" in data


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "total_predictions" in data
    assert "label_distribution" in data
    assert "avg_latency_ms" in data
    assert "p95_latency_ms" in data
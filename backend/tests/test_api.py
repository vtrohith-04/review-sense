"""
Review Sense Backend API Tests
Testing health endpoint, prediction routes, validation, and batch mode.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "active_model" in data
    assert "device" in data


def test_labels_endpoint() -> None:
    response = client.get("/labels")
    assert response.status_code == 200
    data = response.json()
    assert "labels" in data
    assert len(data["labels"]) == 8
    assert "happy" in data["labels"]
    assert "angry" in data["labels"]


def test_single_predict_endpoint() -> None:
    payload = {
        "text": "The delivery was delayed by two weeks and the support agent was rude!",
        "top_k": 3,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert "primary_emotion" in data
    assert "primary_score" in data
    assert "all_scores" in data
    assert len(data["all_scores"]) == 8
    assert "model_used" in data
    assert "latency_ms" in data


def test_batch_predict_endpoint() -> None:
    payload = {
        "texts": [
            "Great product, works as advertised!",
            "Broken item received, extremely angry."
        ],
        "top_k": 2,
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_reviews"] == 2
    assert len(data["results"]) == 2
    assert "total_latency_ms" in data


def test_predict_validation_error() -> None:
    # Empty string should fail Pydantic validation (min_length=1)
    payload = {"text": ""}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


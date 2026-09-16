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


def test_credibility_endpoint() -> None:
    payload = {"text": "AMAZING 10/10 best ever buy it now changed my life holy grail!!!!!"}
    response = client.post("/api/v1/analyze/credibility", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "credibility_score" in data
    assert "fake_probability" in data
    assert "is_fake" in data
    assert "risk_level" in data
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert "flagged_signals" in data
    assert len(data["flagged_signals"]) > 0


def test_batch_credibility_endpoint() -> None:
    payload = {
        "texts": [
            "Works as intended and arrived on time.",
            "BUY NOW 10/10 AMAZING BEST EVER!!!!!"
        ]
    }
    response = client.post("/api/v1/analyze/credibility/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_reviews"] == 2
    assert len(data["results"]) == 2
    assert "total_latency_ms" in data


def test_comprehensive_endpoint() -> None:
    payload = {
        "text": "The delivery was delayed and I am very angry, but customer support was polite.",
        "top_k": 3
    }
    response = client.post("/api/v1/analyze/comprehensive", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "emotion" in data
    assert "credibility" in data
    assert data["emotion"]["primary_emotion"] in ("angry", "frustrated", "sad", "neutral", "happy", "surprised", "fearful", "disgusted")
    assert "credibility_score" in data["credibility"]
    assert "total_latency_ms" in data



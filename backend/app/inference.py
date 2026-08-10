from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from app.config import EMOTION_LABELS


MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "baseline" / "baseline_model.pkl"


class ModelUnavailableError(RuntimeError):
    """Raised when the locally trained baseline artifact is unavailable."""


@lru_cache
def load_baseline_model() -> object:
    if not MODEL_PATH.exists():
        raise ModelUnavailableError(
            "The baseline model is not available. Run 'python ml/scripts/train_baseline.py' first."
        )
    return joblib.load(MODEL_PATH)


def predict_emotions(text: str, threshold: float, top_k: int) -> list[tuple[str, float]]:
    model = load_baseline_model()
    scores = model.predict_proba([text])[0]
    ranked_scores = sorted(zip(EMOTION_LABELS, scores, strict=True), key=lambda item: item[1], reverse=True)
    selected_scores = [item for item in ranked_scores if item[1] >= threshold]

    # Always provide a primary emotion, even when every score is below the caller's threshold.
    if not selected_scores:
        selected_scores = ranked_scores[:1]

    return [(label, float(score)) for label, score in selected_scores[:top_k]]

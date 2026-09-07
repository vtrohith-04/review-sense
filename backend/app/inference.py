"""
Review Sense - Backend Model Inference Engine
Unified predictor loading transformer models with automatic fallback to classical baseline ML.
"""

from __future__ import annotations

import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

# Ensure repository root is in sys.path for ml module imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .config import DEFAULT_THRESHOLD, DEFAULT_TOP_K, EMOTION_LABELS

BASELINE_MODEL_PATH = REPO_ROOT / "ml" / "artifacts" / "baseline" / "baseline_model.pkl"
TRANSFORMER_ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts" / "transformer"


class ModelUnavailableError(RuntimeError):
    """Raised when neither the transformer model nor the baseline ML model is available."""


class InferenceEngineWrapper:
    """Wrapper encapsulating active predictor (Transformer or Baseline TF-IDF)."""

    def __init__(self, predictor: Any, model_name: str, device: str):
        self.predictor = predictor
        self.model_name = model_name
        self.device = device

    def predict_one(
        self,
        text: str,
        top_k: int = DEFAULT_TOP_K,
        threshold_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if self.model_name == "deberta-v3":
            raw_result = self.predictor.predict(
                text=text,
                threshold_override=threshold_override,
                top_k=top_k,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            sec_details = raw_result.get("secondary_details", [])
            secondary_scores = [
                {
                    "label": emotion,
                    "score": round(float(score), 4),
                    "threshold": round(float(thresh), 4),
                    "exceeds_threshold": True,
                }
                for emotion, score, thresh in sec_details[:top_k]
            ]

            return {
                "text": text,
                "primary_emotion": raw_result["primary_emotion"],
                "primary_score": round(float(raw_result["primary_score"]), 4),
                "secondary_emotions": secondary_scores,
                "all_scores": {k: round(float(v), 4) for k, v in raw_result["scores"].items()},
                "model_used": "deberta-v3",
                "latency_ms": round(elapsed_ms, 2),
            }

        # Fallback Baseline Model Strategy (TF-IDF + Logistic Regression)
        scores_arr = self.predictor.predict_proba([text])[0]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        all_scores = {label: round(float(score), 4) for label, score in zip(EMOTION_LABELS, scores_arr)}
        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        primary_emotion, primary_score = ranked[0]
        secondary_scores = []

        for label, score in ranked[1:]:
            thresh = threshold_override.get(label, DEFAULT_THRESHOLD) if threshold_override else DEFAULT_THRESHOLD
            if score >= thresh and len(secondary_scores) < top_k:
                secondary_scores.append({
                    "label": label,
                    "score": score,
                    "threshold": thresh,
                    "exceeds_threshold": True,
                })

        return {
            "text": text,
            "primary_emotion": primary_emotion,
            "primary_score": primary_score,
            "secondary_emotions": secondary_scores,
            "all_scores": all_scores,
            "model_used": "tfidf-baseline",
            "latency_ms": round(elapsed_ms, 2),
        }

    def predict_batch(
        self,
        texts: List[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        results = [self.predict_one(text, top_k=top_k) for text in texts]
        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "total_reviews": len(texts),
            "results": results,
            "total_latency_ms": round(total_elapsed_ms, 2),
        }


@lru_cache(maxsize=1)
def get_inference_engine() -> InferenceEngineWrapper:
    """
    Singleton factory for the active inference engine.
    Attempts loading DeBERTa transformer first; falls back to TF-IDF baseline if unavailable.
    """
    # 1. Attempt loading Transformer model
    if TRANSFORMER_ARTIFACT_DIR.exists():
        try:
            from ml.models.transformer_predictor import TransformerEmotionPredictor
            predictor = TransformerEmotionPredictor(model_dir=TRANSFORMER_ARTIFACT_DIR)
            device_str = str(predictor.device)
            return InferenceEngineWrapper(predictor=predictor, model_name="deberta-v3", device=device_str)
        except Exception as err:
            print(f"[Warning] Could not load DeBERTa transformer predictor: {err}. Falling back to baseline.")

    # 2. Attempt loading TF-IDF Baseline model
    if BASELINE_MODEL_PATH.exists():
        try:
            baseline_model = joblib.load(BASELINE_MODEL_PATH)
            return InferenceEngineWrapper(predictor=baseline_model, model_name="tfidf-baseline", device="cpu")
        except Exception as err:
            raise ModelUnavailableError(f"Failed to load baseline ML model: {err}") from err

    raise ModelUnavailableError(
        "Neither Transformer weights nor baseline ML models were found in ml/artifacts/."
    )


def predict_emotions(
    text: str,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> List[Tuple[str, float]]:
    """Legacy helper function maintained for backwards compatibility."""
    engine = get_inference_engine()
    res = engine.predict_one(text=text, top_k=top_k, threshold_override=None)
    items = [(res["primary_emotion"], res["primary_score"])]
    for sec in res["secondary_emotions"]:
        items.append((sec["label"], sec["score"]))
    return items[:top_k]

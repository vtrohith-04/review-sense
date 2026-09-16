"""
Review Sense - Fake & Spam Review (Credibility Engine) Inference Predictor
Fast, production-ready inference engine for review authenticity scoring and explainable risk detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from scipy.sparse import hstack

from .stylometrics import detect_explainable_signals, extract_stylometric_features


class ReviewCredibilityPredictor:
    """Inference engine for fake review detection and authenticity assessment."""

    def __init__(
        self,
        artifact_dir: Optional[Union[str, Path]] = None,
    ):
        if artifact_dir is None:
            self.artifact_dir = Path(__file__).resolve().parents[1] / "artifacts" / "credibility"
        else:
            self.artifact_dir = Path(artifact_dir)

        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"Credibility artifact directory not found: {self.artifact_dir}")

        clf_path = self.artifact_dir / "credibility_classifier.joblib"
        vec_path = self.artifact_dir / "credibility_vectorizer.joblib"
        scaler_path = self.artifact_dir / "credibility_scaler.joblib"

        if not (clf_path.exists() and vec_path.exists() and scaler_path.exists()):
            raise FileNotFoundError("Missing one or more required credibility model artifacts (.joblib).")

        self.clf = joblib.load(clf_path)
        self.vectorizer = joblib.load(vec_path)
        self.scaler = joblib.load(scaler_path)

    def _calibrate_prediction(
        self,
        raw_fake_prob: float,
        signals: List[str],
        features: Dict[str, float],
    ) -> Tuple[float, float, str]:
        """Blend raw model probability with explainable stylometric risk signals."""
        penalty = 0.0
        # Check signal severity
        if any("Heavy promotional superlatives" in s for s in signals):
            penalty += 0.25
        if any("High punctuation intensity" in s for s in signals):
            penalty += 0.15
        if any("Excessive capitalization" in s for s in signals):
            penalty += 0.15
        if any("Extremely brief polarized review" in s for s in signals):
            penalty += 0.15

        # Blended fake probability
        adjusted_fake_prob = min(max(raw_fake_prob + penalty, 0.0), 1.0)
        credibility_score = 1.0 - adjusted_fake_prob
        risk_level = self._determine_risk_level(adjusted_fake_prob)

        return adjusted_fake_prob, credibility_score, risk_level

    def _determine_risk_level(self, fake_prob: float) -> str:
        if fake_prob >= 0.65:
            return "HIGH"
        elif fake_prob >= 0.40:
            return "MEDIUM"
        else:
            return "LOW"

    def predict_one(self, text: str) -> Dict[str, Any]:
        """Predict credibility score and risk signals for a single review string."""
        if not text or not text.strip():
            return {
                "text": text,
                "is_fake": False,
                "credibility_score": 1.0,
                "fake_probability": 0.0,
                "risk_level": "LOW",
                "flagged_signals": [],
                "stylometrics": extract_stylometric_features(""),
            }

        cleaned = text.strip()
        tfidf_vec = self.vectorizer.transform([cleaned])
        stylo_raw = [extract_stylometric_features(cleaned)]
        stylo_matrix = np.array([[v for v in stylo_raw[0].values()]], dtype=np.float32)
        stylo_scaled = self.scaler.transform(stylo_matrix)

        X = hstack([tfidf_vec, stylo_scaled]).tocsr()
        probs = self.clf.predict_proba(X)[0]
        raw_fake_prob = float(probs[1])

        signals = detect_explainable_signals(cleaned, raw_fake_prob)
        fake_prob, credibility_score, risk_level = self._calibrate_prediction(
            raw_fake_prob, signals, stylo_raw[0]
        )
        is_fake = bool(fake_prob >= 0.50)

        return {
            "text": cleaned,
            "is_fake": is_fake,
            "credibility_score": round(credibility_score, 4),
            "fake_probability": round(fake_prob, 4),
            "risk_level": risk_level,
            "flagged_signals": signals,
            "stylometrics": stylo_raw[0],
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict credibility scores for a list of review texts."""
        if not texts:
            return []

        cleaned_texts = [str(t).strip() if str(t).strip() else "Neutral" for t in texts]
        tfidf_mat = self.vectorizer.transform(cleaned_texts)
        stylo_rows = [extract_stylometric_features(t) for t in cleaned_texts]
        stylo_matrix = np.array([[v for v in r.values()] for r in stylo_rows], dtype=np.float32)
        stylo_scaled = self.scaler.transform(stylo_matrix)

        X = hstack([tfidf_mat, stylo_scaled]).tocsr()
        batch_probs = self.clf.predict_proba(X)

        results = []
        for i, text in enumerate(texts):
            raw_fake_prob = float(batch_probs[i][1])
            signals = detect_explainable_signals(str(text), raw_fake_prob)
            fake_prob, cred_score, risk_level = self._calibrate_prediction(
                raw_fake_prob, signals, stylo_rows[i]
            )
            is_fake = bool(fake_prob >= 0.50)

            results.append({
                "text": str(text),
                "is_fake": is_fake,
                "credibility_score": round(cred_score, 4),
                "fake_probability": round(fake_prob, 4),
                "risk_level": risk_level,
                "flagged_signals": signals,
                "stylometrics": stylo_rows[i],
            })

        return results

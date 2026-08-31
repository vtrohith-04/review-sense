"""
Review Sense - Transformer Emotion Predictor
Fast, calibrated inference engine for multi-label emotion classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_EMOTION_LABELS = [
    "happy",
    "sad",
    "angry",
    "frustrated",
    "surprised",
    "fearful",
    "disgusted",
    "neutral",
]


class TransformerEmotionPredictor:
    """Production inference engine for fine-tuned emotion transformer models."""

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        if model_dir is None:
            self.model_dir = Path(__file__).resolve().parents[1] / "artifacts" / "transformer"
        else:
            self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model artifact directory not found: {self.model_dir}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.max_length = max_length

        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Extract label mapping
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        else:
            self.labels = DEFAULT_EMOTION_LABELS

        # Load calibrated thresholds if available
        thresholds_path = self.model_dir / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, "r", encoding="utf-8") as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = {label: 0.5 for label in self.labels}

    def _format_prediction(
        self,
        probs: np.ndarray,
    ) -> Dict[str, Any]:
        """Format raw probability vector into structured prediction output."""
        prob_dict = {
            label: round(float(probs[i]), 4)
            for i, label in enumerate(self.labels)
        }

        # Primary emotion: class with highest probability
        primary_idx = int(np.argmax(probs))
        primary_emotion = self.labels[primary_idx]
        primary_conf = round(float(probs[primary_idx]), 4)

        # Secondary emotions: any other class exceeding its calibrated threshold
        secondary_emotions = []
        for i, label in enumerate(self.labels):
            if i != primary_idx:
                thresh = self.thresholds.get(label, 0.5)
                if probs[i] >= thresh:
                    secondary_emotions.append(label)

        # Sort secondary emotions by descending probability
        secondary_emotions.sort(key=lambda l: prob_dict[l], reverse=True)

        detected_emotions = [primary_emotion] + [e for e in secondary_emotions if e != primary_emotion]

        return {
            "primary_emotion": primary_emotion,
            "primary_confidence": primary_conf,
            "secondary_emotions": secondary_emotions,
            "detected_emotions": detected_emotions,
            "emotion_probabilities": prob_dict,
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict emotion for a single review text string."""
        if not text or not text.strip():
            # Return neutral fallback for empty strings
            return {
                "primary_emotion": "neutral",
                "primary_confidence": 1.0,
                "secondary_emotions": [],
                "detected_emotions": ["neutral"],
                "emotion_probabilities": {l: (1.0 if l == "neutral" else 0.0) for l in self.labels},
            }

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        return self._format_prediction(probs)

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> List[Dict[str, Any]]:
        """Predict emotions for a batch of review text strings."""
        if not texts:
            return []

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) if str(t).strip() else "neutral" for t in texts[i : i + batch_size]]
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_probs = torch.sigmoid(outputs.logits).cpu().numpy()

            for probs in batch_probs:
                results.append(self._format_prediction(probs))

        return results

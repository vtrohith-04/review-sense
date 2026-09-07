"""
Review Sense - FastAPI Data Schemas
Pydantic v2 schemas for API requests, responses, and validation contracts.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EmotionScore(BaseModel):
    """Score breakdown for an individual emotion label."""
    label: str = Field(..., description="Name of the emotion label (e.g. happy, angry).")
    score: float = Field(..., ge=0.0, le=1.0, description="Predicted probability score between 0.0 and 1.0.")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Calibrated decision threshold for this emotion.")
    exceeds_threshold: bool = Field(..., description="True if score is greater than or equal to threshold.")


class PredictRequest(BaseModel):
    """Payload for single review emotion analysis."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Review text to analyze for emotion content.",
        examples=["The product arrived damaged and customer support was unhelpful!"]
    )
    threshold_override: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional custom threshold overrides for specific emotions."
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Maximum number of secondary emotions to return."
    )


class PredictResponse(BaseModel):
    """Response returned after processing review emotion analysis."""
    text: str = Field(..., description="Original input review text.")
    primary_emotion: str = Field(..., description="Dominant emotion label with highest predicted score.")
    primary_score: float = Field(..., ge=0.0, le=1.0, description="Probability score of primary emotion.")
    secondary_emotions: List[EmotionScore] = Field(
        default_factory=list,
        description="Secondary emotions that crossed calibrated decision thresholds."
    )
    all_scores: Dict[str, float] = Field(
        ...,
        description="Full probability mapping for all 8 target emotions."
    )
    model_used: str = Field(
        ...,
        description="Name of the active inference model engine (e.g. 'deberta-v3' or 'tfidf-baseline')."
    )
    latency_ms: float = Field(..., description="Inference execution time in milliseconds.")


class BatchPredictRequest(BaseModel):
    """Payload for processing multiple review texts in bulk."""
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=500,
        description="List of review strings to classify in batch mode."
    )
    top_k: int = Field(default=3, ge=1, le=8)


class BatchPredictResponse(BaseModel):
    """Response returned for batch review processing."""
    total_reviews: int = Field(..., description="Number of reviews processed.")
    results: List[PredictResponse] = Field(..., description="List of prediction responses.")
    total_latency_ms: float = Field(..., description="Total batch execution time in milliseconds.")


class HealthResponse(BaseModel):
    """Response for backend health check endpoint."""
    status: str = Field(default="ok", description="Service operational status.")
    active_model: str = Field(..., description="Currently active model engine.")
    device: str = Field(..., description="Hardware compute device (cuda or cpu).")
    version: str = Field(default="1.0.0", description="API version string.")

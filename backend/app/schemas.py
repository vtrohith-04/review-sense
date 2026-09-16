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
        min_length=1,
        max_length=500,
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


class CredibilityRequest(BaseModel):
    """Payload for review credibility and spam analysis."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Review text to analyze for credibility/spam.",
        examples=["AMAZING 10/10 best ever buy it now changed my life holy grail!!!!!"]
    )


class CredibilityResponse(BaseModel):
    """Response returned for review credibility and fake detection."""
    text: str = Field(..., description="Original review text.")
    is_fake: bool = Field(..., description="True if review is flagged as likely deceptive or spam.")
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="Authenticity confidence score (1.0 = genuine, 0.0 = fake).")
    fake_probability: float = Field(..., ge=0.0, le=1.0, description="Probability that review is deceptive or spam.")
    risk_level: str = Field(..., description="Authenticity risk tier: LOW, MEDIUM, or HIGH.")
    flagged_signals: List[str] = Field(default_factory=list, description="List of explainable risk signals detected.")
    latency_ms: float = Field(..., description="Execution time in milliseconds.")


class BatchCredibilityRequest(BaseModel):
    """Payload for batch review credibility analysis."""
    texts: List[str] = Field(..., min_length=1, max_length=500, description="List of review strings to analyze.")


class BatchCredibilityResponse(BaseModel):
    """Response for batch credibility analysis."""
    total_reviews: int = Field(..., description="Number of reviews analyzed.")
    results: List[CredibilityResponse] = Field(..., description="List of individual credibility responses.")
    total_latency_ms: float = Field(..., description="Total batch processing latency in milliseconds.")


class ComprehensiveAnalysisRequest(BaseModel):
    """Payload for combined emotion + credibility analysis."""
    text: str = Field(..., min_length=1, max_length=2000, description="Review text to analyze.")
    top_k: int = Field(default=3, ge=1, le=8)


class ComprehensiveAnalysisResponse(BaseModel):
    """Unified response combining 8-class emotion distribution and credibility assessment."""
    text: str = Field(..., description="Original review text.")
    emotion: PredictResponse = Field(..., description="Full multi-label emotion analysis.")
    credibility: CredibilityResponse = Field(..., description="Authenticity and spam risk analysis.")
    total_latency_ms: float = Field(..., description="Total combined execution latency in milliseconds.")



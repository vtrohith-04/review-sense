from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import DEFAULT_THRESHOLD, DEFAULT_TOP_K, EMOTION_LABELS


class HealthResponse(BaseModel):
    status: str
    service: str


class EmotionPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review or feedback text to analyze.")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=3)
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class EmotionScore(BaseModel):
    label: str
    score: float


class EmotionPredictionResponse(BaseModel):
    primary_emotion: str
    top_emotions: list[EmotionScore]
    model_name: str
    mode: str


app = FastAPI(
    title="Review Sense API",
    version="0.1.0",
    description="Backend scaffold for review emotion classification.",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service="review-sense-api")


@app.get("/labels")
def get_labels() -> dict[str, list[str]]:
    return {"labels": EMOTION_LABELS}


@app.post("/api/v1/predict/emotion", response_model=EmotionPredictionResponse)
def predict_emotion(payload: EmotionPredictionRequest) -> EmotionPredictionResponse:
    # Placeholder response until the ML inference pipeline is connected.
    top_emotions = [
        EmotionScore(label="neutral", score=0.51),
        EmotionScore(label="happy", score=0.27),
        EmotionScore(label="frustrated", score=0.22),
    ][: payload.top_k]

    return EmotionPredictionResponse(
        primary_emotion=top_emotions[0].label,
        top_emotions=top_emotions,
        model_name="placeholder-baseline",
        mode="multi_label_stub",
    )

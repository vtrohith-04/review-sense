from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import DEFAULT_THRESHOLD, DEFAULT_TOP_K, EMOTION_LABELS
from app.inference import ModelUnavailableError, predict_emotions


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
    description="API for Review Sense emotion classification.",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service="review-sense-api")


@app.get("/labels")
def get_labels() -> dict[str, list[str]]:
    return {"labels": EMOTION_LABELS}


@app.post("/api/v1/predict/emotion", response_model=EmotionPredictionResponse)
def predict_emotion(payload: EmotionPredictionRequest) -> EmotionPredictionResponse:
    try:
        predictions = predict_emotions(payload.text, payload.threshold, payload.top_k)
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    top_emotions = [EmotionScore(label=label, score=round(score, 4)) for label, score in predictions]

    return EmotionPredictionResponse(
        primary_emotion=top_emotions[0].label,
        top_emotions=top_emotions,
        model_name="tfidf-logistic-regression-baseline",
        mode="multi_label",
    )

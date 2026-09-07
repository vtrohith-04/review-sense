"""
Review Sense - FastAPI Backend Service
Serving fine-grained multi-label emotion prediction endpoints.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .config import EMOTION_LABELS
from .inference import ModelUnavailableError, get_inference_engine
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


app = FastAPI(
    title="Review Sense API",
    version="1.0.0",
    description="Production REST API for multi-label review emotion classification.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS) for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Returns system status, active inference model, and hardware device."""
    try:
        engine = get_inference_engine()
        return HealthResponse(
            status="ok",
            active_model=engine.model_name,
            device=engine.device,
            version="1.0.0",
        )
    except ModelUnavailableError as err:
        return HealthResponse(
            status="degraded",
            active_model="none",
            device="cpu",
            version="1.0.0",
        )


@app.get("/labels", tags=["Metadata"])
def get_labels() -> dict[str, list[str]]:
    """Returns the list of 8 target emotion labels."""
    return {"labels": EMOTION_LABELS}


@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_single_review(payload: PredictRequest) -> PredictResponse:
    """
    Classify fine-grained emotions for a single review text.
    Returns primary emotion, secondary emotions, full probability distribution, and latency.
    """
    try:
        engine = get_inference_engine()
        result_dict = engine.predict_one(
            text=payload.text,
            top_k=payload.top_k,
            threshold_override=payload.threshold_override,
        )
        return PredictResponse(**result_dict)
    except ModelUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing error: {err}",
        ) from err


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_batch_reviews(payload: BatchPredictRequest) -> BatchPredictResponse:
    """
    Classify fine-grained emotions for a list of review texts in batch mode.
    """
    try:
        engine = get_inference_engine()
        result_dict = engine.predict_batch(
            texts=payload.texts,
            top_k=payload.top_k,
        )
        return BatchPredictResponse(**result_dict)
    except ModelUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        ) from err


# Legacy endpoint maintained for backwards compatibility with earlier UI prototypes
@app.post("/api/v1/predict/emotion", tags=["Legacy"])
def legacy_predict_emotion(payload: PredictRequest) -> PredictResponse:
    return predict_single_review(payload)

"""
Review Sense - FastAPI Backend Service
Serving fine-grained multi-label emotion prediction endpoints.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .config import EMOTION_LABELS
from .inference import (
    ModelUnavailableError,
    get_credibility_engine,
    get_inference_engine,
)
from .schemas import (
    BatchCredibilityRequest,
    BatchCredibilityResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    ComprehensiveAnalysisRequest,
    ComprehensiveAnalysisResponse,
    CredibilityRequest,
    CredibilityResponse,
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


@app.post(
    "/api/v1/analyze/credibility",
    response_model=CredibilityResponse,
    status_code=status.HTTP_200_OK,
    tags=["Credibility & Spam"],
)
def analyze_review_credibility(payload: CredibilityRequest) -> CredibilityResponse:
    """
    Evaluate review authenticity and flag potential fake, bot, or promotional spam.
    Returns credibility score (0.0–1.0), risk level (LOW/MEDIUM/HIGH), and explainable signals.
    """
    try:
        engine = get_credibility_engine()
        result_dict = engine.analyze_one(text=payload.text)
        return CredibilityResponse(**result_dict)
    except ModelUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Credibility analysis error: {err}",
        ) from err


@app.post(
    "/api/v1/analyze/credibility/batch",
    response_model=BatchCredibilityResponse,
    status_code=status.HTTP_200_OK,
    tags=["Credibility & Spam"],
)
def analyze_batch_credibility(payload: BatchCredibilityRequest) -> BatchCredibilityResponse:
    """
    Batch evaluate review authenticity and spam risk for multiple reviews.
    """
    try:
        engine = get_credibility_engine()
        result_dict = engine.analyze_batch(texts=payload.texts)
        return BatchCredibilityResponse(**result_dict)
    except ModelUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        ) from err


@app.post(
    "/api/v1/analyze/comprehensive",
    response_model=ComprehensiveAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Comprehensive"],
)
def analyze_comprehensive(payload: ComprehensiveAnalysisRequest) -> ComprehensiveAnalysisResponse:
    """
    Combined endpoint: performs both 8-class multi-label emotion prediction
    AND authenticity / spam credibility assessment in a single request.
    """
    start_time = time.perf_counter()
    try:
        emotion_engine = get_inference_engine()
        cred_engine = get_credibility_engine()

        emotion_res = emotion_engine.predict_one(text=payload.text, top_k=payload.top_k)
        cred_res = cred_engine.analyze_one(text=payload.text)
        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ComprehensiveAnalysisResponse(
            text=payload.text,
            emotion=PredictResponse(**emotion_res),
            credibility=CredibilityResponse(**cred_res),
            total_latency_ms=round(total_elapsed_ms, 2),
        )
    except ModelUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comprehensive analysis error: {err}",
        ) from err


# Legacy endpoint maintained for backwards compatibility with earlier UI prototypes
@app.post("/api/v1/predict/emotion", tags=["Legacy"])
def legacy_predict_emotion(payload: PredictRequest) -> PredictResponse:
    return predict_single_review(payload)


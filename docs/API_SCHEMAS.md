# Review Sense — API Schemas & Endpoint Specifications

## 1. Overview

The **Review Sense** REST API is built with **FastAPI** and **Pydantic v2**. All request and response bodies use strict JSON formatting. 

Interactive OpenAPI documentation is automatically available when running the backend service at `http://localhost:8000/docs`.

---

## 2. API Endpoints Summary

| Method | Endpoint | Description | Request Body | Response Body |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health check & active model status | None | `HealthResponse` |
| `POST` | `/api/v1/predict` | Single review emotion prediction | `PredictRequest` | `PredictResponse` |
| `POST` | `/api/v1/predict/batch` | Bulk review emotion prediction | `BatchPredictRequest` | `BatchPredictResponse` |

---

## 3. Detailed Request & Response Contracts

### 3.1 Endpoint: `GET /health`

**Response Example (`200 OK`):**
```json
{
  "status": "ok",
  "active_model": "deberta-v3",
  "device": "cuda",
  "version": "1.0.0"
}
```

---

### 3.2 Endpoint: `POST /api/v1/predict`

**Request Body (`PredictRequest`):**
```json
{
  "text": "The delivery was delayed by a week and customer support ignored my emails!",
  "top_k": 3
}
```

**Response Body (`PredictResponse` - `200 OK`):**
```json
{
  "text": "The delivery was delayed by a week and customer support ignored my emails!",
  "primary_emotion": "angry",
  "primary_score": 0.8421,
  "secondary_emotions": [
    {
      "label": "frustrated",
      "score": 0.5210,
      "threshold": 0.3300,
      "exceeds_threshold": true
    }
  ],
  "all_scores": {
    "happy": 0.0102,
    "sad": 0.1540,
    "angry": 0.8421,
    "frustrated": 0.5210,
    "surprised": 0.0412,
    "fearful": 0.0215,
    "disgusted": 0.1105,
    "neutral": 0.0321
  },
  "model_used": "deberta-v3",
  "latency_ms": 28.45
}
```

---

### 3.3 Endpoint: `POST /api/v1/predict/batch`

**Request Body (`BatchPredictRequest`):**
```json
{
  "texts": [
    "Love this product, high quality!",
    "Item arrived broken on arrival."
  ],
  "top_k": 2
}
```

**Response Body (`BatchPredictResponse` - `200 OK`):**
```json
{
  "total_reviews": 2,
  "results": [
    {
      "text": "Love this product, high quality!",
      "primary_emotion": "happy",
      "primary_score": 0.9512,
      "secondary_emotions": [],
      "all_scores": { "happy": 0.9512, "neutral": 0.0488 },
      "model_used": "deberta-v3",
      "latency_ms": 14.12
    },
    {
      "text": "Item arrived broken on arrival.",
      "primary_emotion": "angry",
      "primary_score": 0.7812,
      "secondary_emotions": [
        {
          "label": "sad",
          "score": 0.6410,
          "threshold": 0.6200,
          "exceeds_threshold": true
        }
      ],
      "all_scores": { "angry": 0.7812, "sad": 0.6410 },
      "model_used": "deberta-v3",
      "latency_ms": 15.34
    }
  ],
  "total_latency_ms": 29.46
}
```

---

## 4. HTTP Error Handling & Codes

* `400 Bad Request`: Payload validation failed (e.g. text exceeds 2,000 characters or batch exceeds 500 items).
* `422 Unprocessable Entity`: Invalid JSON payload structure.
* `503 Service Unavailable`: Model weights missing and baseline failed to load.


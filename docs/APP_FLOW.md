# Review Sense — App Flow & System Workflow Specification

## 1. System Sequence Diagram (Single Text Analysis)

The diagram below illustrates the end-to-end data flow when a user analyzes a review in the frontend dashboard:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React Dashboard
    participant API as FastAPI Backend
    participant Predictor as Transformer Predictor
    participant DeBERTa as DeBERTa-v3 Model (PyTorch)
    participant Calibrator as Threshold Engine

    User->>UI: Types review text & clicks "Analyze Emotion"
    UI->>UI: Validates input length (1-2000 chars)
    UI->>API: POST /api/v1/predict { "text": "Item arrived damaged, very unhappy!" }
    API->>API: Pydantic validates PredictRequest schema
    API->>Predictor: predict(text, top_k=3)
    Predictor->>DeBERTa: Tokenize & forward pass (logits)
    DeBERTa-->>Predictor: Raw Logits z_1..z_8
    Predictor->>Calibrator: Apply Sigmoid & compare against thresholds.json
    Calibrator-->>Predictor: Primary: "angry" (0.84), Secondary: ["frustrated" (0.52)]
    Predictor-->>API: Struct { primary_emotion, secondary_emotions, all_scores, latency_ms }
    API-->>UI: 200 OK JSON Response
    UI->>UI: Render Emotion Hero Card & Animated Probability Bar Charts
```

---

## 2. Model Selection & Fallback Workflow

When the backend service starts, it runs an automatic model availability check to pick the best available inference engine:

```mermaid
flowchart TD
    Start([FastAPI Startup]) --> CheckDir{Does ml/artifacts/transformer/ exist?}
    CheckDir -- Yes --> CheckWeights{Are model.safetensors or pytorch_model.bin present?}
    CheckDir -- No --> Fallback[Load TF-IDF + Logistic Regression Baseline]
    CheckWeights -- Yes --> InitDeBERTa[Initialize TransformerEmotionPredictor<br/>Device: GPU/CUDA if available, else CPU]
    CheckWeights -- No --> Fallback
    InitDeBERTa --> ReadyDeBERTa[Set ACTIVE_MODEL = 'deberta-v3']
    Fallback --> ReadyBaseline[Set ACTIVE_MODEL = 'tfidf-baseline']
    ReadyDeBERTa --> Serve([Service Ready on /api/v1/predict])
    ReadyBaseline --> Serve
```

---

## 3. Batch Review CSV Processing Flow

For bulk review processing (e.g. e-commerce feedback uploads):

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React Dashboard
    participant API as FastAPI Backend
    participant Predictor as Transformer Predictor

    User->>UI: Drops reviews.csv into Batch Upload Modal
    UI->>UI: Client-side CSV parse & extracts "review_text" column
    UI->>API: POST /api/v1/predict/batch { "texts": [...] }
    API->>Predictor: predict_batch(texts)
    Predictor->>Predictor: Tokenize & Batch Forward Pass (PyTorch DataLoader)
    Predictor-->>API: BatchPredictionResults [...]
    API-->>UI: 200 OK JSON Batch Response
    UI->>UI: Render Aggregated Emotion Breakdown Pie Chart & Export CSV Button
```

---

## 4. User Journeys & State Transitions

### Primary Single Review Journey
1. **Initial State:** User lands on dashboard; active model indicator shows `DeBERTa-v3 (Calibrated)`.
2. **Input State:** User enters text or clicks a preset sample chip (*"Fast delivery but the quality is disappointing"*).
3. **Loading State:** Input button shows loading spinner; previous charts fade out.
4. **Success State:** Hero badge displays primary emotion (`sad`), secondary badge displays (`frustrated`), and 8 animated progress bars show exact score percentages.
5. **Error State:** If API returns validation error (e.g. empty string), a red alert banner displays the issue cleanly.


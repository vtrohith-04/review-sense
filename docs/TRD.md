# Review Sense — Technical Requirements Document (TRD)

## 1. System Architecture Overview

The **Review Sense** platform follows a decoupled multi-tier architecture:
- **ML Engine (`ml/`):** Transformer fine-tuning, threshold calibration, and production inference engine ([transformer_predictor.py](file:///d:/Downloads/review-sense.worktrees/phase-4-completion-analysis-next-steps/ml/models/transformer_predictor.py)).
- **Backend API (`backend/`):** High-performance REST service powered by **FastAPI** and **Pydantic v2**.
- **Frontend UI (`frontend/`):** Interactive dashboard built with **React**, **TypeScript**, **Vite**, and **TailwindCSS**.

---

## 2. Machine Learning Architecture & Pipeline

### 2.1 Model Taxonomy
The core classifier maps text input to an 8-class emotion distribution:
1. `happy`
2. `sad`
3. `angry`
4. `frustrated`
5. `surprised`
6. `fearful`
7. `disgusted`
8. `neutral`

### 2.2 Model Architecture (DeBERTa-v3)
* **Base Model:** `microsoft/deberta-v3-base` (12 layers, 768 hidden size, 86M parameters).
* **Multi-Label Head:** Linear layer outputting 8 raw logits ($z_1, z_2, \dots, z_8$).
* **Loss Function:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) to support multi-label probabilities:
  $$\mathcal{L} = -\frac{1}{C} \sum_{c=1}^{C} \left[ y_c \log(\sigma(z_c)) + (1 - y_c) \log(1 - \sigma(z_c)) \right]$$
* **Inference Activation:** Sigmoid $\sigma(z_c) = \frac{1}{1 + e^{-z_c}}$ applied independently per emotion logit.

### 2.3 Threshold Calibration Strategy
Instead of relying on a static $0.50$ cutoff across all classes, decision thresholds are empirically calibrated on validation data to maximize F1-score per label:

```json
{
  "happy": 0.47,
  "sad": 0.62,
  "angry": 0.29,
  "frustrated": 0.33,
  "surprised": 0.36,
  "fearful": 0.22,
  "disgusted": 0.39,
  "neutral": 0.22
}
```

* **Primary Emotion:** $\text{Label } c \text{ where } \sigma(z_c) = \max_i(\sigma(z_i))$.
* **Secondary Emotions:** All labels $c \neq \text{Primary}$ where $\sigma(z_c) \ge \text{Threshold}_c$.

---

## 3. Serving Engine & Backend Architecture

### 3.1 Model Loading & Memory Strategy
* **Singleton Predictor:** Model weights (`model.safetensors` or local weights) and tokenizers are loaded lazily on application startup using `@lru_cache` to eliminate per-request loading overhead.
* **Device Allocation:** Automatic GPU detection (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) with FP16 precision support.

### 3.2 Resilience & Graceful Degradation
If the transformer artifact directory (`ml/artifacts/transformer/`) is missing or CUDA memory is depleted:
1. Catch `FileNotFoundError` / `RuntimeError`.
2. Fallback seamlessly to the Phase 3 TF-IDF + Logistic Regression model ([baseline_model.pkl](file:///d:/Downloads/review-sense.worktrees/phase-4-completion-analysis-next-steps/ml/artifacts/baseline/baseline_model.pkl)).
3. Tag the API response `model_used: "tfidf-baseline"` so clients remain informed.

---

## 4. Performance & Scalability Specs

* **Concurrency:** FastAPI async handlers handle concurrent non-blocking requests.
* **Batch Vectorization:** Transformer tokenizer uses dynamic padding (`padding=True, truncation=True, max_length=128`) for efficient GPU batch inference.
* **Memory Footprint:** DeBERTa-v3 inference requires ~1.2 GB RAM (CPU) or ~1.5 GB VRAM (GPU).

---

## 5. Security & Validation

* **Input Sanitization:** Strips control characters and enforces length boundaries ($1 \le \text{length} \le 2000$).
* **CORS Policy:** Allowed origins configured via `backend/app/config.py`.
* **Data Schemas:** All incoming requests and outgoing responses conform strictly to Pydantic v2 schemas.


# Review Sense — Product Requirements Document (PRD)

## 1. Executive Summary & Vision

**Review Sense** is an AI-powered customer feedback intelligence platform designed to classify fine-grained emotion nuances in product reviews, support tickets, and customer survey data.

Traditional sentiment analysis tools categorize text into basic binary or 3-class buckets (*Positive*, *Negative*, *Neutral*). While helpful, this coarse categorization fails to provide actionable insights for product and customer experience teams. A negative review could stem from **anger** (product failure), **frustration** (shipping delay), **disgust** (quality defect), or **sadness** (disappointed expectations). 

Review Sense bridges this gap by serving an 8-class multi-label emotion classification engine powered by state-of-the-art fine-tuned Transformer models (**DeBERTa-v3**) with calibrated decision thresholds.

---

## 2. Problem Statement

1. **Coarse Sentiment Limits Actionability:** "Negative" sentiment does not specify what operational team should handle the issue (Logistics vs Engineering vs Support).
2. **Multi-Label Reality:** Real customer reviews often convey multiple overlapping emotions simultaneously (e.g., *surprised* by a defect and *angry* about poor customer service).
3. **Threshold Ambiguity:** Standard 0.50 classification cutoffs over-predict dominant classes while suppressing critical minority emotions like *fearful* or *disgusted*.

---

## 3. Target User Personas

### Persona A: E-Commerce Product Manager
* **Goal:** Monitor post-launch product feedback to identify top feature requests or quality complaints.
* **Pain Point:** Reading thousands of raw reviews manually or getting generic "60% Negative" sentiment reports.
* **Needs:** Granular emotion breakdown (`frustrated` vs `angry`) and bulk CSV review processing.

### Persona B: Customer Experience & Support Lead
* **Goal:** Route urgent customer support tickets to specialized agents.
* **Pain Point:** High-priority angry customers wait in the same queue as general inquiries.
* **Needs:** Low-latency API (<150ms) to flag `angry` or `frustrated` reviews instantly upon submission.

---

## 4. Functional Requirements (FRs)

| ID | Feature Name | Description | Priority |
| :--- | :--- | :--- | :--- |
| **FR-1** | Real-Time Single Text Analysis | Accepts text input (up to 2,000 chars) and predicts primary emotion + secondary emotions. | High |
| **FR-2** | 8-Emotion Multi-Label Scoring | Outputs probability scores ($0.0 - 1.0$) for all 8 emotions (`happy`, `sad`, `angry`, `frustrated`, `surprised`, `fearful`, `disgusted`, `neutral`). | High |
| **FR-3** | Calibrated Thresholding | Filters secondary emotions using validation-tuned decision thresholds rather than a generic 0.50 cutoff. | High |
| **FR-4** | Batch Review Upload | Accepts `.csv` files containing up to 500 reviews, processing them in bulk with aggregated emotion analytics. | Medium |
| **FR-5** | Model Transparency & Fallback | Displays active model engine (`DeBERTa-v3` vs `TF-IDF Baseline`) and inference execution latency in milliseconds. | Medium |
| **FR-6** | Fake / Spam Review Flag (Milestone 2) | Second classification head to predict authenticity score and flag suspicious review patterns. | Future |

---

## 5. Non-Functional Requirements (NFRs)

### Performance & Latency
* **Single Prediction SLA:** $\le 150\text{ ms}$ on GPU / $\le 350\text{ ms}$ on CPU.
* **Batch Processing SLA:** $\le 5\text{ seconds}$ for 100 review texts.

### Reliability & Resilience
* **Model Fallback:** If DeBERTa transformer weights are missing or GPU memory is depleted, the backend must gracefully degrade to the cached TF-IDF + Logistic Regression baseline without throwing 500 errors.
* **Validation Hardening:** 100% request payload validation via Pydantic v2 schemas.

### Maintainability & Standards
* Clean separation between ML inference layer (`ml/`), FastAPI backend (`backend/`), and React UI (`frontend/`).

---

## 6. Success Metrics & KPIs

* **Classification Accuracy:** DeBERTa-v3 Micro F1 $\ge 0.67$, Macro F1 $\ge 0.60$ on held-out test split.
* **API Uptime:** 99.9% availability for `/predict` and `/health` endpoints.
* **User Experience:** Instant visual feedback in React UI with animated emotion probability bar charts.


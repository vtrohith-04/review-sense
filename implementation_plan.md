# Review Sense Implementation Plan

## Project Goal

Build Review Sense into a real NLP project that classifies emotions in review-like text using both:

- a classical ML baseline (TF-IDF + Logistic Regression)
- a fine-tuned transformer model (DistilBERT Multi-Label Classifier)

The first milestone is an overall review emotion classifier with primary and secondary emotion detection. The second milestone is fake/spam review detection.

---

## Final Emotion Labels (8-Class Taxonomy)

The working production label set:

- `happy`
- `sad`
- `angry`
- `frustrated`
- `surprised`
- `fearful`
- `disgusted`
- `neutral`

---

## Model Strategy

### Baseline (Phase 3 - Completed)
- TF-IDF vectorization (30K max features, unigram + bigram, min_df=2)
- Logistic Regression (OneVsRestClassifier, class_weight='balanced')
- Benchmark: Test Micro F1 = 0.5936, Macro F1 = 0.5240, Subset Accuracy = 40.54%

### Deep Learning Model (Phase 4 - In Progress)
- **Base Architecture**: `microsoft/deberta-v3-base` with sequence classification head (8 labels) for maximum classification accuracy (utilizing disentangled attention and replaced token detection)
- **Loss Function**: `BCEWithLogitsLoss` for multi-label classification
- **Training Config**: 3 epochs, AdamW optimizer, lr=2e-5 with linear warmup, mixed precision (`fp16`) on NVIDIA RTX 4060 GPU
- **Threshold Strategy**: Validation-tuned thresholds per emotion label to balance precision/recall for both frequent (`happy`, `neutral`) and rare (`disgusted`, `fearful`, `surprised`, `frustrated`) classes.

### Inference & Output Schema (For Phase 5 API & Phase 6 Frontend)
- Return raw probability scores for all 8 emotions: `{ happy: 0.85, surprised: 0.42, ... }`
- **Primary Emotion**: Class with the highest probability score.
- **Secondary Emotions**: Additional classes whose probabilities exceed their respective tuned thresholds, ordered by confidence.
- Fast, CPU/GPU compatible inference pipeline exportable to FastAPI backend.

---

## Dataset Strategy

### Dataset: GoEmotions (Mapped)
- Source: Google GoEmotions (58k Reddit comments with 27 fine-grained emotions + neutral)
- Mapped to the 8 production emotions
- Splits:
  - **Train**: 40,700 samples
  - **Validation**: 5,100 samples
  - **Test**: 5,079 samples

---

## Roadmap & Phase Status

### Phase 1: Foundation ✅
- Finalize 8-emotion taxonomy
- Structure repository (`ml/`, `backend/`, `frontend/`, `docs/`)
- Document architecture and mapping notes

### Phase 2: Dataset Pipeline ✅
- Fetch GoEmotions dataset
- Build mapping from 27 labels to 8 business-friendly labels
- Save processed train/val/test CSV splits (`ml/data/processed/goemotions_mapped/`)

### Phase 3: Baseline ML & Error Review ✅
- Train TF-IDF + Logistic Regression baseline
- Comprehensive confusion-focused error analysis (`PHASE_3_COMPLETION_REPORT.md`)
- Establish baseline benchmark metrics (Micro F1: 0.5936, Macro F1: 0.5240)

### Phase 4: Deep NLP (DistilBERT Transformer) 🚀 [CURRENT]
- Dataset tokenization with `DistilBertTokenizerFast`
- Fine-tune `DistilBertForSequenceClassification` multi-label model on GPU
- Multi-label threshold optimization on validation set
- Multi-label inference engine for primary and secondary emotion prediction
- Comprehensive evaluation on test set and comparative report against Phase 3 baseline
- Export trained model artifacts to `ml/artifacts/transformer/`

### Phase 5: API Layer & Service Integration ⏳ [NEXT]
- FastAPI application in `backend/`
- Load Phase 4 fine-tuned DistilBERT artifacts (model, tokenizer, threshold config)
- Endpoints:
  - `GET /health`: Service health and model metadata
  - `POST /predict`: Single review emotion classification (primary emotion, secondary emotions, full probability distribution)
  - `POST /predict/batch`: Batch review classification with analytics summary
- Fast inference latency optimization and fallback support for baseline model

### Phase 6: Frontend Redesign & Analytics Dashboard ⏳
- Modern React / Next.js or interactive UI in `frontend/`
- Real-time review input and interactive emotion radar / bar chart
- Batch CSV review upload and emotional distribution analytics
- Display primary vs secondary emotions with confidence indicators

### Phase 7: Spam & Fake Review Detection ⏳
- Dataset selection for fake / deceptive review detection
- Binary / Multi-task classifier training
- Dedicated API endpoint `POST /analyze/credibility`

### Phase 8: Deployment & GitHub Polish ⏳
- Docker containerization for backend and frontend
- README polish with architecture diagrams, demo GIFs, and benchmark tables
- Final submission readiness


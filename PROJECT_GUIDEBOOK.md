# Review Sense Project Guidebook

This is the living project guide for Review Sense. It is designed to help you recall the project, understand where it stands, and review key metrics without opening the codebase.

- Project name: Review Sense
- Repository: vtrohith-04/review-sense
- Current status: Phase 3 complete, Phase 4 in planning
- Last updated: August 29, 2026

## 1. Executive Summary

Review Sense is an NLP project focused on reading short review-like text and classifying the dominant emotion behind it. The project is built to evolve from a research prototype into a polished AI product that can support review intelligence workflows.

The project follows a practical, portfolio-friendly structure:

- Classical ML baseline: TF-IDF + Logistic Regression
- Deep learning model: DistilBERT fine-tuned for multi-label emotion classification
- Backend API: FastAPI service for inference
- Frontend: React + TypeScript prototype for review input and analytics
- Expansion path: fake/spam review detection and domain-adapted review analysis

The first objective is not just "sentiment analysis". The main value is detecting nuanced emotions such as happy, sad, angry, frustrated, surprised, fearful, disgusted, and neutral.

## 2. Mission and Problem Statement

The core problem is this:

- product reviews and customer feedback contain emotional context beyond simple positive/negative sentiment
- businesses need to understand how a reviewer feels, not just whether the review is good or bad
- a model that produces a primary emotion and a few secondary emotions provides a richer, more useful summary

The project intends to turn review text into structured emotional insights such as:

- primary emotion
- top 2 to 3 supporting emotions
- confidence scores
- summary of repeated review patterns

## 3. Product Direction

The project is framed as a review intelligence platform with emotion analysis as the first product layer. Later phases can extend it to include:

- fake/spam review detection
- review-domain adaptation
- business dashboards
- anomaly detection on customer sentiment patterns

This makes the project useful for both academic evaluation and portfolio presentation.

## 4. Final Label Set

We are using a practical 8-label taxonomy:

- happy
- sad
- angry
- frustrated
- surprised
- fearful
- disgusted
- neutral

This taxonomy is easier to explain than a 27-label emotion taxonomy and is better suited for product demos, stakeholder conversations, and frontend UI design.

## 5. Why This Architecture

### 5.1 Classical baseline first

A TF-IDF + Logistic Regression model is useful because it establishes a reliable baseline.

Benefits:

- fast to train and debug
- easy to explain to non-technical stakeholders
- good benchmark for measuring transformer improvements
- useful as a fallback or comparison model

### 5.2 Transformer model for real understanding

DistilBERT is used to capture context, semantics, and subtle wording patterns that a bag-of-words model cannot.

Benefits:

- higher performance on nuanced emotional language
- better handling of short text and implicit emotion
- stronger academic and portfolio value

### 5.3 Hybrid inference strategy

At inference time:

- the highest score becomes the primary emotion
- the top 2 to 3 emotions above threshold are surfaced as secondary signals
- this creates a more realistic product behavior than a single label hard prediction

## 6. Dataset Strategy

### Primary dataset: GoEmotions

GoEmotions is the main benchmark dataset used to train and evaluate the project.

Key facts:

- source: public English text dataset from Google Research
- domain: short text comments, mostly Reddit-like feedback
- original taxonomy: 27 emotions + neutral
- strength: suited for emotion classification, not only sentiment classification

Why it fits the project:

- public and credible dataset
- widely discussed in NLP research
- supports multi-label emotion tasks
- provides a strong academic foundation

### Important limitation

GoEmotions is not a pure product review dataset. It is best described as:

> a general review and feedback emotion classifier trained on a strong public emotion benchmark, with future adaptation to product-review-specific data.

### Label mapping logic

The raw label space is reduced into a business-friendly 8-emotion mapping:

- happy: joy, amusement, approval, gratitude, optimism, love, caring, desire, excitement, pride, relief
- sad: sadness, grief, disappointment, remorse
- angry: anger, annoyance
- frustrated: disapproval, confusion, embarrassment
- surprised: surprise, realization, curiosity
- fearful: fear, nervousness
- disgusted: disgust
- neutral: neutral

This keeps the model interpretable without sacrificing meaningful nuance.

## 7. Project Architecture

### 7.1 Repository layout

```text
review-sense/
  backend/
    app/
      api/
      config.py
      main.py
  docs/
    implementation-phases.md
    interview-notes.md
    dataset-notes.md
  frontend/
    src/
    README.md
  ml/
    artifacts/
    scripts/
    requirements.txt
  README.md
  PROJECT_GUIDEBOOK.md
  PHASE_3_SUMMARY.md
  ERROR_ANALYSIS_README.md
  PHASE_3_COMPLETION_REPORT.md
```

### 7.2 Layered design

- Frontend: user interface, input, batch review handling, result display
- Backend: inference API, validation, schema, request/response structure
- ML layer: training pipeline, evaluation, model artifact generation
- Documentation layer: phase reports, dataset notes, project guide

## 8. Technology Stack

### Frontend

- React
- TypeScript
- Vite

### Backend

- FastAPI
- Pydantic
- Uvicorn

### ML / Data Science

- Python
- pandas
- numpy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- evaluate

## 9. Project Timeline

### Phase 1: Repo foundation

- separate prototype frontend from ML code
- create backend and ML folders
- freeze emotion taxonomy
- document architecture and goals

### Phase 2: Dataset and label mapping

- integrate GoEmotions
- map raw labels to the final 8 labels
- define dataset preprocessing workflow

### Phase 3: Baseline NLP

- TF-IDF + Logistic Regression
- benchmark performance
- generate confusion-focused error review
- save metrics and visual artifacts

### Phase 4: Deep NLP

- DistilBERT fine-tuning
- multi-label inference
- threshold tuning for secondary emotions
- compare against baseline

### Phase 5: API and UX

- FastAPI inference service
- frontend redesign
- batch upload and analytics

### Phase 6: Expansion

- spam review detection
- review-domain adaptation
- deployment polish

## 10. Current Phase Status

### Phase 3 status: complete

This project has completed the baseline NLP phase with a full evaluation package.

### Model used

- TF-IDF + Logistic Regression (OneVsRest)
- max features: 30K
- feature style: bigrams included
- class balancing: weighted training

### Results snapshot

| Metric | Validation | Test |
|---|---:|---:|
| Micro F1 | 0.5913 | 0.5936 |
| Macro F1 | 0.5099 | 0.5240 |
| Subset Accuracy | 41.00% | 40.54% |

### Key findings

- model generalizes well; validation and test metrics are close
- happy and neutral are among the strongest labels
- frustrated, surprised, and disgusted are weaker and harder to distinguish
- class imbalance and semantic overlap are the main sources of error
- the TF-IDF baseline is useful but limited in context understanding

## 11. Phase 3 Metrics and Artifacts

Artifacts are saved under:

- `ml/artifacts/baseline/error_analysis/`

### Visuals

- [Confusion Matrix](./ml/artifacts/baseline/error_analysis/confusion_matrix.png)
- [Per Label Metrics](./ml/artifacts/baseline/error_analysis/per_label_metrics.png)
- [Error Rates](./ml/artifacts/baseline/error_analysis/error_rates.png)

### Data reports

- `summary.json`
- `confusion_report.json`
- `misclassifications.json`

### Error analysis highlights

Top weak emotions:

1. frustrated
2. surprised
3. disgusted

Best-performing emotions:

1. happy
2. neutral
3. fearful

Common confusion patterns:

- Frustrated → Sad
- Angry → Neutral
- Happy → Angry
- Surprised → Surprised (self-confusion)
- Sad → Sad (self-confusion)

## 12. Why Phase 3 Matters

Phase 3 established the minimum reliable benchmark needed before moving to deeper transformer-based work.

It answers three important questions:

1. Is the problem learnable with a strong baseline?
2. What labels are easy or hard?
3. Where should the next model focus its improvements?

This is the foundation for Phase 4, where transformer fine-tuning should improve weak classes and semantic nuance.

## 13. Risks and Learnings

### Major risks

- rare emotion classes are underrepresented
- GoEmotions labels are not identical to product review labels
- TF-IDF struggles with subtle emotional wording
- some emotions overlap semantically (angry vs frustrated)

### Learnings

- a baseline benchmark is essential before model comparison
- error analysis is as important as raw scores
- business-friendly emotion labels help communication and demos
- the future product should emphasize primary + secondary emotions instead of only a single label

## 14. Recommended Next Phase

The immediate next step is Phase 4: transformer-based deep NLP.

High-priority tasks:

- implement DistilBERT fine-tuning
- modify training to support multi-label prediction
- tune thresholds per emotion
- compare transformer metrics against the baseline
- inspect whether rare classes improve under the deep model

## 15. Frontend and Product Vision

The frontend is currently a prototype that demonstrates the idea of review emotion analysis but is not yet fully aligned with the final AI-product flow.

Planned frontend tasks:

- replace direct Gemini-style demo calls with backend API calls
- redesign UI into a cleaner AI product layout
- show primary emotion and top emotion chips
- display confidence scores and supporting emotional signals
- add batch upload and analytics views

## 16. Backend Vision

The backend should become the central inference service for this project.

Expected responsibilities:

- validate incoming review text
- run model inference
- return primary emotion, secondary emotions, and confidence values
- support batch requests
- expose separation between analytics jobs and prediction endpoints

## 17. Project Goals at a Glance

- classify emotion in short review-like text
- generate primary + secondary emotion predictions
- build a strong academic ML pipeline
- compare baseline and transformer model performance
- produce an interface that makes insights understandable
- position the project for portfolio and interview credibility

## 18. How to Update This Guide After Each Phase

After every major project milestone, update this document in the same format:

1. Update the date and current phase status
2. Add a short summary of what was built
3. Record the exact metrics achieved
4. List key files and artifacts created
5. Add screenshots or image links
6. Record lessons learned and next risks
7. State the next phase objective clearly

Suggested update block:

```markdown
### Phase X Update
- Status: Complete / In Progress
- Date:
- Deliverables:
- Metrics:
- Key artifacts:
- Lessons learned:
- Next focus:
```

This makes the document a true living guide instead of a static README.

## 19. Quick Reference

- Main project summary: `README.md`
- Phase plan: `docs/implementation-phases.md`
- Dataset notes: `docs/dataset-notes.md`
- Interview framing: `docs/interview-notes.md`
- Phase 3 summary: `PHASE_3_SUMMARY.md`
- Phase 3 full report: `PHASE_3_COMPLETION_REPORT.md`
- Error analysis quick note: `ERROR_ANALYSIS_README.md`
- Living guide: `PROJECT_GUIDEBOOK.md`

## 20. Final Note

This project is already beyond a simple demo. It has a clear research direction, a practical taxonomy, a modern ML pipeline, a measurable baseline, and a path toward a polished review-intelligence product.

The guidebook is meant to serve as a memory aid and project reference. When a new phase completes, it should be updated so the repository remains understandable even without reading all code.

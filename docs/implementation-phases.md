# Review Sense Phases

## Phase 1: Repo Foundation

- separate prototype frontend from future ML code
- create backend and ML directories
- freeze emotion taxonomy

## Phase 2: Dataset and Label Mapping

- integrate GoEmotions
- map raw labels to the final 8 labels
- document why certain labels merge into business-friendly names

## Phase 3: Baseline NLP

- TF-IDF plus Logistic Regression
- benchmark performance
- create confusion-focused error review

## Phase 4: Deep NLP

- DistilBERT fine-tuning
- multi-label inference
- threshold tuning for secondary emotions

## Phase 5: API and UX

- FastAPI service
- frontend redesign
- batch upload and analytics

## Phase 6: Expansion

- spam review detection
- review-domain adaptation
- deployment and GitHub polish

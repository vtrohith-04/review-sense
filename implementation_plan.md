# Review Sense Implementation Plan

## Project Goal

Build Review Sense into a real NLP project that classifies emotions in review-like text using both:

- a classical ML baseline
- a fine-tuned transformer model

The first milestone is an overall review emotion classifier. The second milestone is fake/spam review detection.

## Final Emotion Labels

The working production label set is:

- `happy`
- `sad`
- `angry`
- `frustrated`
- `surprised`
- `fearful`
- `disgusted`
- `neutral`

## Model Strategy

### Baseline

- TF-IDF vectorization
- Logistic Regression

### Deep Learning Model

- DistilBERT
- multi-label classification head

### Inference Strategy

- return scores for all emotions
- pick highest score as primary emotion
- return top 2 to 3 emotions above threshold as secondary emotions

## Dataset Strategy

### First Dataset

- GoEmotions

### Why It Works

- strong public benchmark
- supports emotion understanding well
- suitable for multi-label learning

### Caveat

- not a pure product review dataset
- project should be described as a general review and feedback emotion classifier in early phases

## Phases

### Phase 1: Foundation

- finalize labels
- structure repo
- document architecture

### Phase 2: Dataset Pipeline

- fetch dataset
- inspect classes
- design mapping from source labels to target labels
- preprocess text

### Phase 3: Baseline ML

- train TF-IDF plus Logistic Regression
- evaluate
- save metrics

### Phase 4: Transformer Model

- tokenize data
- fine-tune DistilBERT
- evaluate and compare
- export checkpoint

### Phase 5: API Layer

- implement FastAPI service
- create prediction schema
- create health endpoint

### Phase 6: Frontend Upgrade

- redesign current UI
- connect to backend
- render probabilities and top emotions

### Phase 7: Fake Review Detection

- choose second dataset
- train second classifier
- expose second endpoint

### Phase 8: GitHub and Submission Readiness

- improve README
- add screenshots
- add architecture diagram
- create first public-ready commit


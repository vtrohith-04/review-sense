# Review Sense

Review Sense is an NLP-focused emotion analysis project for reviews and other short feedback text. The goal is to evolve this repository from a frontend prototype into a complete ML and DL system with a real dataset, training pipeline, evaluation workflow, backend API, and polished frontend.

## What This Project Is

- **Primary domain:** Natural Language Processing
- **Core task:** Multi-label emotion classification for review-like text
- **Current academic framing:** College project with strong portfolio quality
- **Planned product direction:** Review intelligence platform with emotion analysis first and fake/spam review detection next

This project will support hybrid behavior:

- A review gets one **primary emotion** based on the highest model score.
- A review can also expose **top 2 to 3 emotions** when multiple emotions cross a confidence threshold.

## Final Emotion Set

We are using a practical, interview-friendly 8-emotion label set:

- `happy`
- `sad`
- `angry`
- `frustrated`
- `surprised`
- `fearful`
- `disgusted`
- `neutral`

These labels are easier to explain to faculty, reviewers, and future stakeholders than more abstract affect taxonomies, while still giving enough range for meaningful analysis.

## Why ML Baseline Plus DL Model Comparison

This project will include:

1. A classical **ML baseline** using TF-IDF plus Logistic Regression.
2. A **DL model** using DistilBERT fine-tuned for multi-label emotion classification.

We are doing both because:

- it demonstrates understanding of classical and modern NLP
- it creates a measurable benchmark instead of using DL by default
- it gives stronger academic and portfolio value
- it lets us compare accuracy, macro-F1, inference speed, and interpretability

## Dataset Choice

The recommended starting dataset is **GoEmotions**.

### What GoEmotions Is

GoEmotions is a public text emotion dataset released by Google Research. It contains short English text samples annotated with emotion labels and is widely used for emotion classification experiments.

### Why We Chose It

- it is already labeled for emotion understanding
- it supports multi-label setups better than ordinary sentiment datasets
- it is well known enough to discuss confidently in interviews
- it helps us build a rigorous NLP pipeline before doing product-review-specific fine-tuning

### Limitation to Explain in Interviews

GoEmotions is not a pure product review dataset. So the first production-quality milestone is best described as:

> an overall review and feedback emotion classifier built on a robust public emotion dataset, with later adaptation for product reviews

That is a valid and strong explanation for academic review, interviews, and stakeholder discussions.

## Tech Stack

### Frontend

- React
- TypeScript
- Vite

### Backend

- FastAPI
- Pydantic
- Uvicorn

### ML and DL

- pandas
- numpy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- evaluate

## Repository Layout

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
  frontend/
    src/
      components/
      services/
  ml/
    data/
    models/
    reports/
    scripts/
  .gitignore
  README.md
```

## Phase-by-Phase Plan

### Phase 1

- clean project structure
- freeze label taxonomy
- document architecture
- prepare frontend, backend, and ML folders

### Phase 2

- load and inspect GoEmotions
- preprocess data
- map raw labels into the final 8-emotion scheme
- produce train, validation, and test splits

### Phase 3

- build TF-IDF plus Logistic Regression baseline
- evaluate with precision, recall, macro-F1, and confusion insights
- save baseline artifacts

### Phase 4

- fine-tune DistilBERT for multi-label classification
- compute probabilities for each emotion
- derive primary emotion and top 2 to 3 emotions
- compare results against the baseline

### Phase 5

- expose the trained model through FastAPI
- create prediction endpoints
- add structured request and response schemas

### Phase 6

- redesign the frontend into a stronger AI-product style interface
- connect it to the backend API
- show primary emotion, top emotions, and confidence scores

### Phase 7

- add fake/spam review classification
- train a second model or pipeline
- extend the API and UI

### Phase 8

- polish documentation
- add screenshots and architecture diagrams
- initialize Git and push to GitHub

## Why FastAPI, PyTorch, and Hugging Face

### FastAPI

- cleaner validation than Flask
- strong API ergonomics
- good for structured ML inference services

### PyTorch

- widely used in modern NLP
- flexible for experimentation and fine-tuning

### Hugging Face

- easiest practical route for transformer-based NLP
- supports datasets, tokenizers, pretrained models, and evaluation workflows

## Windows-First Development

This repo is being structured for Windows-first local development. Later we can add containerization and deployment support for broader environments.

## Current Status

- Frontend prototype separated into `frontend/`
- Backend scaffold created in `backend/`
- ML workspace scaffold created in `ml/`
- Documentation rewritten for the real NLP direction

## Next Recommended Step

The next build step is:

1. add the GoEmotions dataset loader
2. define label mapping into the final 8 emotions
3. train the baseline ML classifier


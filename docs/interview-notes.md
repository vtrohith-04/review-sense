# Interview Notes

## How to Describe the Project

Review Sense is an NLP project that classifies emotions in review-like text. It uses a classical ML baseline and a transformer-based deep learning model so that we can compare simple bag-of-words methods against contextual language understanding.

## Why Not DL Only

- the ML baseline proves the transformer adds value
- comparisons make the project more rigorous
- it shows both breadth and depth of NLP knowledge

## Why GoEmotions

- public and credible dataset
- suitable for emotion classification
- supports multi-label style tasks
- fast way to build a strong first version

## Limitation to Mention

GoEmotions is not a pure product review dataset, so the first milestone is better described as a general review and feedback emotion classifier. Later phases can adapt the model to product-review-specific data.

## Why FastAPI

- strong validation with Pydantic
- clean inference endpoints
- better structure than a minimal Flask app for ML APIs

## Why DistilBERT

- lighter than BERT
- faster to train and infer
- strong enough for a first major project

## Hybrid Prediction Explanation

The model is trained in a multi-label setup. At inference time, the highest score becomes the primary emotion, and any other emotions above a threshold are shown as secondary emotions.

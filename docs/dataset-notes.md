# Dataset Notes

## Primary Dataset: GoEmotions

GoEmotions is a public English emotion dataset introduced by Google Research for fine-grained emotion classification.

### Why It Fits Review Sense

- it is strong for emotion understanding, not just positive/negative sentiment
- it supports multi-label reasoning, which matches the hybrid output we want
- it is widely recognized in NLP work, so it is easy to justify in college reviews and interviews

### Core Facts

- language: English
- source domain: Reddit comments
- original taxonomy: 27 emotions plus `neutral`
- use case: fine-grained emotion classification

### Important Interview Note

The paper describes roughly 58k carefully curated comments. On Hugging Face, you may also see raw or expanded representations of the dataset depending on the config you load. For this project, that is not a contradiction; it reflects different published views of the same benchmark data.

The safe way to explain it is:

> GoEmotions is a public benchmark for emotion classification. It contains English short-text comments labeled with 27 emotions plus neutral, and it is commonly accessed through Hugging Face with different dataset configurations.

### Why We Are Not Using It As-Is

The original label space is too fine-grained for the first version of Review Sense. For a first major project, the business-friendly 8-emotion space is easier to:

- interpret
- visualize
- explain in demos
- use in the frontend

### Our Target Emotion Mapping

- `happy`: joy, amusement, approval, gratitude, optimism, love, caring, desire, excitement, pride, relief
- `sad`: sadness, grief, disappointment, remorse
- `angry`: anger, annoyance
- `frustrated`: disapproval, confusion, embarrassment
- `surprised`: surprise, realization, curiosity
- `fearful`: fear, nervousness
- `disgusted`: disgust
- `neutral`: neutral

### Limitation

GoEmotions is not a pure product-review dataset. That means:

- it is enough for Phase 1 NLP model building
- it is not the final dataset strategy for a production-grade review classifier

Later, we can add:

- product review datasets for domain adaptation
- fake/spam review datasets for the second model

## Can You Download It and Inspect It?

Yes. That is a good idea.

For this project, the best workflow is:

1. load it through the `datasets` Python library
2. inspect sample rows, splits, and labels
3. save a small local summary for reference
4. create our mapped 8-emotion version for training

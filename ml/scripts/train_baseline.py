from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from label_config import EMOTION_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TF-IDF + logistic regression baseline for Review Sense."
    )
    parser.add_argument(
        "--data-dir",
        default="ml/data/processed/goemotions_mapped",
        help="Directory containing train/validation/test CSV files from prepare_goemotions_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/artifacts/baseline",
        help="Directory where the trained model and evaluation metrics will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for converting multi-label scores into binary predictions.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30000,
        help="Maximum vocabulary size for the TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF features.",
    )
    return parser.parse_args()


def load_split(data_dir: Path, split_name: str) -> pd.DataFrame:
    split_path = data_dir / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_path}. Run ml/scripts/prepare_goemotions_dataset.py first."
        )

    df = pd.read_csv(split_path)
    required_columns = {"text", *EMOTION_LABELS}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Split {split_name} is missing required columns: {missing_columns}")

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    return df


def build_pipeline(max_features: int, min_df: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    )
                ),
            ),
        ]
    )


def predict_with_threshold(model: Pipeline, texts: pd.Series, threshold: float) -> pd.DataFrame:
    probabilities = model.predict_proba(texts)
    predictions = (probabilities >= threshold).astype(int)

    # Keep at least one positive label so downstream demos always have a primary emotion.
    for row_index, row in enumerate(predictions):
        if row.sum() == 0:
            top_label_index = int(probabilities[row_index].argmax())
            row[top_label_index] = 1

    return pd.DataFrame(predictions, columns=EMOTION_LABELS, index=texts.index)


def evaluate_split(model: Pipeline, df: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    y_true = df[EMOTION_LABELS].astype(int)
    y_pred = predict_with_threshold(model, df["text"], threshold)

    return {
        "rows": int(len(df)),
        "subset_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 4),
        "micro_precision": round(float(precision_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "micro_recall": round(float(recall_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(data_dir, "train")
    validation_df = load_split(data_dir, "validation")
    test_df = load_split(data_dir, "test")

    model = build_pipeline(max_features=args.max_features, min_df=args.min_df)
    model.fit(train_df["text"], train_df[EMOTION_LABELS].astype(int))

    metrics = {
        "label_schema": EMOTION_LABELS,
        "threshold": args.threshold,
        "train_rows": int(len(train_df)),
        "validation": evaluate_split(model, validation_df, args.threshold),
        "test": evaluate_split(model, test_df, args.threshold),
    }

    model_path = output_dir / "baseline_model.pkl"
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    metrics_path = output_dir / "metrics.json"
    save_json(metrics_path, metrics)

    print(f"Saved baseline model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

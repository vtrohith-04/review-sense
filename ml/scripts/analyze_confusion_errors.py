"""
Confusion-focused error analysis for Review Sense baseline model.

This script:
1. Loads the trained baseline model and test data
2. Generates predictions and identifies misclassifications
3. Analyzes confusion patterns between emotion labels
4. Creates a detailed error report with:
   - Confusion matrix by emotion label
   - Per-label precision, recall, and F1 with error rates
   - Top misclassification patterns
   - Sample misclassifications with text excerpts
   - Confusion heatmap visualization
5. Saves all reports and visualizations for review
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from label_config import EMOTION_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze confusion patterns and errors in the baseline model."
    )
    parser.add_argument(
        "--model-path",
        default="ml/artifacts/baseline/baseline_model.pkl",
        help="Path to the trained baseline model.",
    )
    parser.add_argument(
        "--test-data-path",
        default="ml/data/processed/goemotions_mapped/test.csv",
        help="Path to the test dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/artifacts/baseline/error_analysis",
        help="Directory where error analysis reports and visualizations will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for converting multi-label scores into binary predictions.",
    )
    parser.add_argument(
        "--top-n-samples",
        type=int,
        default=10,
        help="Number of top misclassification samples to report per pattern.",
    )
    return parser.parse_args()


def load_model(model_path: Path) -> Any:
    """Load the trained baseline model from pickle."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with model_path.open("rb") as f:
        return pickle.load(f)


def load_test_data(test_data_path: Path) -> pd.DataFrame:
    """Load and validate test data."""
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    df = pd.read_csv(test_data_path)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    return df


def predict_with_threshold(model: Any, texts: pd.Series, threshold: float) -> pd.DataFrame:
    """Generate predictions with a threshold."""
    probabilities = model.predict_proba(texts)
    predictions = (probabilities >= threshold).astype(int)

    for row_index, row in enumerate(predictions):
        if row.sum() == 0:
            top_label_index = int(probabilities[row_index].argmax())
            row[top_label_index] = 1

    return pd.DataFrame(predictions, columns=EMOTION_LABELS, index=texts.index)


def get_probabilities(model: Any, texts: pd.Series) -> pd.DataFrame:
    """Get raw prediction probabilities."""
    probabilities = model.predict_proba(texts)
    return pd.DataFrame(probabilities, columns=EMOTION_LABELS, index=texts.index)


def generate_confusion_matrix_report(
    y_true: pd.DataFrame, y_pred: pd.DataFrame
) -> dict[str, Any]:
    """Generate per-label confusion metrics."""
    report = {}

    for label in EMOTION_LABELS:
        true_col = y_true[label]
        pred_col = y_pred[label]

        # Calculate confusion elements
        tp = ((true_col == 1) & (pred_col == 1)).sum()
        fp = ((true_col == 0) & (pred_col == 1)).sum()
        fn = ((true_col == 1) & (pred_col == 0)).sum()
        tn = ((true_col == 0) & (pred_col == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Error rates
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        report[label] = {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "false_positive_rate": round(float(false_positive_rate), 4),
            "false_negative_rate": round(float(false_negative_rate), 4),
            "support": int(true_col.sum()),
        }

    return report


def find_misclassified_samples(
    test_df: pd.DataFrame,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    probabilities: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """
    Find misclassified samples and group by confusion pattern.
    A confusion pattern is: {true_label -> predicted_label}
    """
    misclassifications: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx in range(len(test_df)):
        true_labels = set(EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS)) if y_true.iloc[idx, i])
        pred_labels = set(EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS)) if y_pred.iloc[idx, i])

        if true_labels != pred_labels:
            # Get primary emotion (highest probability)
            pred_probs = probabilities.iloc[idx]
            primary_pred = pred_probs.idxmax()
            primary_pred_prob = float(pred_probs.max())

            primary_true = EMOTION_LABELS[max(range(len(EMOTION_LABELS)), key=lambda i: y_true.iloc[idx, i])]

            # Create a confusion pattern key
            confusion_key = f"{primary_true} -> {primary_pred}"

            misclassifications[confusion_key].append(
                {
                    "text": test_df.iloc[idx]["text"],
                    "true_labels": sorted(true_labels),
                    "predicted_labels": sorted(pred_labels),
                    "primary_true": primary_true,
                    "primary_pred": primary_pred,
                    "primary_pred_prob": round(primary_pred_prob, 4),
                    "index": int(idx),
                }
            )

    # Sort each confusion pattern by confidence (probability) descending
    for pattern in misclassifications:
        misclassifications[pattern].sort(key=lambda x: x["primary_pred_prob"], reverse=True)
        misclassifications[pattern] = misclassifications[pattern][:top_n]

    # Sort patterns by frequency
    sorted_patterns = sorted(misclassifications.items(), key=lambda x: len(x[1]), reverse=True)
    return dict(sorted_patterns)


def generate_confusion_matrix_numeric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> np.ndarray:
    """Generate numeric confusion matrix for heatmap visualization."""
    n_labels = len(EMOTION_LABELS)
    cm = np.zeros((n_labels, n_labels))

    for true_idx in range(n_labels):
        for pred_idx in range(n_labels):
            true_col = y_true.iloc[:, true_idx]
            pred_col = y_pred.iloc[:, pred_idx]

            # Count cases where true_label[i] = 1 and pred_label[j] = 1
            cm[true_idx, pred_idx] = ((true_col == 1) & (pred_col == 1)).sum()

    return cm


def create_confusion_heatmap(
    cm: np.ndarray, output_path: Path, figsize: tuple[int, int] = (12, 10)
) -> None:
    """Create and save confusion matrix heatmap."""
    # Ensure matrix is integer type for proper formatting
    cm = cm.astype(int)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cbar_kws={"label": "Number of Samples"},
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix: True Labels vs Predicted Labels")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_per_label_metrics_plot(confusion_report: dict[str, Any], output_path: Path) -> None:
    """Create bar plot comparing metrics across emotion labels."""
    labels = list(confusion_report.keys())
    f1_scores = [confusion_report[label]["f1"] for label in labels]
    precisions = [confusion_report[label]["precision"] for label in labels]
    recalls = [confusion_report[label]["recall"] for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precisions, width, label="Precision", alpha=0.8)
    ax.bar(x, recalls, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1_scores, width, label="F1", alpha=0.8)

    ax.set_xlabel("Emotion Label")
    ax.set_ylabel("Score")
    ax.set_title("Per-Label Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_error_rate_plot(confusion_report: dict[str, Any], output_path: Path) -> None:
    """Create plot showing false positive and false negative rates."""
    labels = list(confusion_report.keys())
    fp_rates = [confusion_report[label]["false_positive_rate"] for label in labels]
    fn_rates = [confusion_report[label]["false_negative_rate"] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, fp_rates, width, label="False Positive Rate", alpha=0.8, color="salmon")
    ax.bar(x + width / 2, fn_rates, width, label="False Negative Rate", alpha=0.8, color="lightcoral")

    ax.set_xlabel("Emotion Label")
    ax.set_ylabel("Rate")
    ax.set_title("Error Rates by Emotion Label")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_json(path: Path, payload: dict | Any) -> None:
    """Save a dictionary or object as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def generate_summary_report(
    confusion_report: dict[str, Any],
    misclassifications: dict[str, list[dict[str, Any]]],
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a high-level summary of confusion and errors."""
    total_samples = len(y_true)
    total_mismatches = sum(1 for i in range(total_samples) if not (y_true.iloc[i] == y_pred.iloc[i]).all())
    mismatch_rate = total_mismatches / total_samples if total_samples > 0 else 0

    # Identify most confused labels
    worst_f1_labels = sorted(confusion_report.items(), key=lambda x: x[1]["f1"])[:3]
    worst_recall_labels = sorted(confusion_report.items(), key=lambda x: x[1]["recall"])[:3]
    worst_precision_labels = sorted(confusion_report.items(), key=lambda x: x[1]["precision"])[:3]

    # Top confusion patterns
    top_patterns = list(misclassifications.items())[:5]

    summary = {
        "total_test_samples": total_samples,
        "misclassified_samples": total_mismatches,
        "mismatch_rate": round(float(mismatch_rate), 4),
        "worst_f1_scores": {label: metrics["f1"] for label, metrics in worst_f1_labels},
        "worst_recall_scores": {label: metrics["recall"] for label, metrics in worst_recall_labels},
        "worst_precision_scores": {label: metrics["precision"] for label, metrics in worst_precision_labels},
        "top_confusion_patterns": [
            {
                "pattern": pattern,
                "count": len(samples),
            }
            for pattern, samples in top_patterns
        ],
    }

    return summary


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    test_data_path = Path(args.test_data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and test data...")
    model = load_model(model_path)
    test_df = load_test_data(test_data_path)

    print("Generating predictions...")
    y_true = test_df[EMOTION_LABELS].astype(int)
    y_pred = predict_with_threshold(model, test_df["text"], args.threshold)
    probabilities = get_probabilities(model, test_df["text"])

    print("Analyzing confusion patterns...")
    confusion_report = generate_confusion_matrix_report(y_true, y_pred)
    misclassifications = find_misclassified_samples(test_df, y_true, y_pred, probabilities, args.top_n_samples)
    cm_numeric = generate_confusion_matrix_numeric(y_true, y_pred)

    print("Generating summary report...")
    summary = generate_summary_report(confusion_report, misclassifications, y_true, y_pred)

    print("Creating visualizations...")
    create_confusion_heatmap(cm_numeric, output_dir / "confusion_matrix.png")
    create_per_label_metrics_plot(confusion_report, output_dir / "per_label_metrics.png")
    create_error_rate_plot(confusion_report, output_dir / "error_rates.png")

    print("Saving reports...")
    save_json(output_dir / "confusion_report.json", confusion_report)
    save_json(output_dir / "misclassifications.json", misclassifications)
    save_json(output_dir / "summary.json", summary)

    print(f"\n{'='*80}")
    print("CONFUSION-FOCUSED ERROR REVIEW SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total test samples: {summary['total_test_samples']}")
    print(f"Misclassified samples: {summary['misclassified_samples']}")
    print(f"Mismatch rate: {summary['mismatch_rate']:.2%}\n")

    print("Top Confusion Patterns:")
    for pattern_data in summary["top_confusion_patterns"]:
        print(f"  {pattern_data['pattern']}: {pattern_data['count']} occurrences")

    print("\nLabels with Lowest F1 Scores:")
    for label, f1 in summary["worst_f1_scores"].items():
        recall = confusion_report[label]["recall"]
        precision = confusion_report[label]["precision"]
        print(f"  {label:12} -> F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    print("\n" + "="*80)
    print(f"Error analysis saved to: {output_dir}")
    print("Generated files:")
    print(f"  - confusion_matrix.png (heatmap visualization)")
    print(f"  - per_label_metrics.png (precision/recall/F1 comparison)")
    print(f"  - error_rates.png (false positive/negative rates)")
    print(f"  - confusion_report.json (detailed per-label metrics)")
    print(f"  - misclassifications.json (sample errors grouped by pattern)")
    print(f"  - summary.json (high-level error summary)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

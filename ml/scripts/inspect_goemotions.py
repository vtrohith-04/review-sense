from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_dataset

from label_config import GOEMOTIONS_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the GoEmotions dataset and save a local summary."
    )
    parser.add_argument(
        "--config",
        default="raw",
        choices=["raw", "simplified"],
        help="Hugging Face dataset config to load.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of sample rows to save from each split.",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/reports/goemotions",
        help="Directory where summaries and samples will be written.",
    )
    return parser.parse_args()


def get_split_summary(split_name: str, dataset_split) -> dict:
    row = dataset_split[0]
    summary = {
        "split": split_name,
        "rows": len(dataset_split),
        "columns": list(dataset_split.column_names),
    }

    if "text" in row:
        text_lengths = [len(text) for text in dataset_split.select(range(min(500, len(dataset_split))))["text"]]
        summary["avg_text_length_first_500"] = round(sum(text_lengths) / len(text_lengths), 2)
        summary["min_text_length_first_500"] = min(text_lengths)
        summary["max_text_length_first_500"] = max(text_lengths)

    label_columns = [column for column in GOEMOTIONS_LABELS if column in dataset_split.column_names]
    if label_columns:
        label_totals = {}
        for label in label_columns:
            label_totals[label] = int(sum(dataset_split[label]))
        summary["label_totals"] = label_totals
    elif "labels" in dataset_split.column_names:
        summary["labels_feature"] = str(dataset_split.features["labels"])

    return summary


def save_sample_csv(split_name: str, dataset_split, output_dir: Path, sample_size: int) -> None:
    sample_size = min(sample_size, len(dataset_split))
    sample_rows = dataset_split.select(range(sample_size)).to_pandas()
    sample_rows.to_csv(output_dir / f"{split_name}_sample.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset: DatasetDict = load_dataset("google-research-datasets/go_emotions", args.config)

    summary = {
        "dataset": "google-research-datasets/go_emotions",
        "config": args.config,
        "splits": {},
    }

    for split_name, dataset_split in dataset.items():
        summary["splits"][split_name] = get_split_summary(split_name, dataset_split)
        save_sample_csv(split_name, dataset_split, output_dir, args.sample_size)

    summary_path = output_dir / f"{args.config}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved dataset summary to: {summary_path}")
    print(f"Saved sample files under: {output_dir}")


if __name__ == "__main__":
    main()

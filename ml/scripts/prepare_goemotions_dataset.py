from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_dataset

from label_config import GOEMOTIONS_LABELS, GOEMOTIONS_MAPPING_NOTES


TARGET_LABELS = list(GOEMOTIONS_MAPPING_NOTES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map GoEmotions labels into the Review Sense 8-emotion schema."
    )
    parser.add_argument(
        "--config",
        default="simplified",
        choices=["raw", "simplified"],
        help="Hugging Face GoEmotions config to load (default: simplified, with train/validation/test splits).",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/data/processed/goemotions_mapped",
        help="Output directory for mapped CSV files.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep rows where no target emotion is activated after mapping.",
    )
    return parser.parse_args()


def normalize_label_ids(value: object) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [int(label_id) for label_id in value]
    if isinstance(value, tuple):
        return [int(label_id) for label_id in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [int(label_id) for label_id in stripped.split(",") if label_id.strip()]
    if pd.isna(value):
        return []
    return [int(value)]


def build_source_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    if set(GOEMOTIONS_LABELS).issubset(df.columns):
        return df[GOEMOTIONS_LABELS].fillna(0).astype(int)

    if "labels" not in df.columns:
        raise ValueError(
            "Dataset split is missing GoEmotions label columns. Expected either raw one-hot columns "
            "or a 'labels' column with emotion ids."
        )

    records: list[dict[str, int]] = []
    for label_ids in df["labels"]:
        active_ids = set(normalize_label_ids(label_ids))
        records.append(
            {
                label_name: int(label_index in active_ids)
                for label_index, label_name in enumerate(GOEMOTIONS_LABELS)
            }
        )
    return pd.DataFrame(records, index=df.index)


def map_row_to_target_labels(row: pd.Series) -> dict[str, int]:
    mapped = {}
    for target_label, source_labels in GOEMOTIONS_MAPPING_NOTES.items():
        mapped[target_label] = int(any(int(row.get(source_label, 0)) == 1 for source_label in source_labels))
    return mapped


def prepare_split(df: pd.DataFrame, keep_empty: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    source_label_frame = build_source_label_frame(df)
    mapped_labels = source_label_frame.apply(map_row_to_target_labels, axis=1, result_type="expand")

    base_columns = [column for column in ["text", "id"] if column in df.columns]
    prepared = pd.concat([df[base_columns].copy(), mapped_labels], axis=1)
    prepared["source_label_count"] = source_label_frame.sum(axis=1)
    prepared["emotion_count"] = prepared[TARGET_LABELS].sum(axis=1)
    prepared["has_target_emotion"] = prepared["emotion_count"].gt(0)
    prepared["primary_emotion"] = prepared[TARGET_LABELS].idxmax(axis=1).where(
        prepared["has_target_emotion"],
        pd.NA,
    )

    dropped_empty_rows = int((~prepared["has_target_emotion"]).sum())
    if not keep_empty:
        prepared = prepared[prepared["has_target_emotion"]].reset_index(drop=True)

    split_summary = {
        "input_rows": int(len(df)),
        "kept_rows": int(len(prepared)),
        "dropped_empty_rows": 0 if keep_empty else dropped_empty_rows,
        "rows_without_target_emotion": dropped_empty_rows,
    }
    return prepared, split_summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset: DatasetDict = load_dataset("google-research-datasets/go_emotions", args.config)
    summary: dict[str, object] = {
        "dataset": "google-research-datasets/go_emotions",
        "config": args.config,
        "target_labels": TARGET_LABELS,
        "keep_empty": args.keep_empty,
        "splits": {},
    }

    for split_name, dataset_split in dataset.items():
        df = dataset_split.to_pandas()
        prepared, split_summary = prepare_split(df, keep_empty=args.keep_empty)
        output_path = output_dir / f"{split_name}.csv"
        prepared.to_csv(output_path, index=False)
        split_summary["target_label_totals"] = {
            label: int(prepared[label].sum()) for label in TARGET_LABELS
        }
        summary["splits"][split_name] = split_summary
        print(f"Saved mapped split: {output_path}")

    summary_path = output_dir / "mapping_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved mapping summary: {summary_path}")


if __name__ == "__main__":
    main()

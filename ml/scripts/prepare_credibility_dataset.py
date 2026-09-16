#!/usr/bin/env python3
"""
Review Sense - Fake & Spam Review Dataset Preparation Pipeline
Loads the 40k Fake/Authentic Reviews Benchmark dataset, cleans, stratifies,
and saves train/val/test splits to ml/data/processed/credibility/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_credibility_dataset")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "ml" / "data" / "processed" / "credibility"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading 'theArijitDas/Fake-Reviews-Dataset' from Hugging Face...")
    ds = datasets.load_dataset("theArijitDas/Fake-Reviews-Dataset", split="train")
    df = pd.DataFrame(ds)

    logger.info(f"Loaded raw dataset with {len(df)} rows.")

    # Data hygiene
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 3].copy()
    df = df.drop_duplicates(subset=["text"]).copy()

    # Standardize schema: text, label (0=Authentic, 1=Fake/Spam), category, rating
    df["label"] = df["label"].astype(int)
    df["is_fake"] = df["label"]
    df["authenticity"] = df["label"].map({0: "authentic", 1: "fake_or_spam"})

    logger.info(f"Cleaned dataset: {len(df)} samples.")
    logger.info(f"Class distribution:\n{df['authenticity'].value_counts()}")

    # Stratified Train (80%) / Val (10%) / Test (10%) Split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    logger.info(f"Train set: {len(train_df)} samples ({train_df['label'].value_counts().to_dict()})")
    logger.info(f"Val set:   {len(val_df)} samples ({val_df['label'].value_counts().to_dict()})")
    logger.info(f"Test set:  {len(test_df)} samples ({test_df['label'].value_counts().to_dict()})")

    # Save CSVs
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    summary = {
        "dataset_name": "theArijitDas/Fake-Reviews-Dataset",
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "class_mapping": {
            0: "authentic",
            1: "fake_or_spam"
        },
        "class_counts": df["authenticity"].value_counts().to_dict(),
        "categories": df["category"].value_counts().to_dict() if "category" in df else {}
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved processed dataset splits to {output_dir}")


if __name__ == "__main__":
    main()

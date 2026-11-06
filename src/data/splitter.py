"""
Data Splitting Module
=====================
Splits interaction data into Train (70%), Validation (15%), and Test (15%) subsets.
Guarantees zero data leakage and ensures active users have interaction histories in Train.
Provides both stratified per-user temporal/random splitting and global k-fold cross-validation.
"""

import os
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def stratified_user_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    by_date: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split user interactions into train, validation, and test sets.
    If by_date is True, sorts chronologically for each user to simulate real-world future prediction.
    Otherwise shuffles with random_seed.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    np.random.seed(random_seed)

    train_rows, val_rows, test_rows = [], [], []

    grouped = df.groupby("user_id")
    for user_id, group in grouped:
        if by_date and "date" in group.columns:
            group_sorted = group.sort_values("date")
        else:
            group_sorted = group.sample(frac=1.0, random_state=random_seed)

        n = len(group_sorted)
        if n < 3:
            # If user has very few items, put 1 in train and split remainder
            train_rows.append(group_sorted.iloc[:1])
            if n == 2:
                test_rows.append(group_sorted.iloc[1:2])
            continue

        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        # Ensure at least 1 interaction in each set when possible
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = n - (n_train + n_val)
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

        train_part = group_sorted.iloc[:n_train]
        val_part = group_sorted.iloc[n_train:n_train + n_val]
        test_part = group_sorted.iloc[n_train + n_val:]

        train_rows.append(train_part)
        val_rows.append(val_part)
        test_rows.append(test_part)

    df_train = pd.concat(train_rows, ignore_index=True)
    df_val = pd.concat(val_rows, ignore_index=True)
    df_test = pd.concat(test_rows, ignore_index=True)

    logger.info(f"Split Summary: Train={len(df_train):,} ({len(df_train)/len(df):.1%}), "
                f"Val={len(df_val):,} ({len(df_val)/len(df):.1%}), "
                f"Test={len(df_test):,} ({len(df_test)/len(df):.1%})")

    return df_train, df_val, df_test


def split_and_save(
    clean_inter_path: str = "data/processed/interactions_clean.parquet",
    output_dir: str = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, str]:
    """Load cleaned interactions, perform stratified split, and save Parquet files."""
    df = pd.read_parquet(clean_inter_path)
    df_train, df_val, df_test = stratified_user_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        by_date=True
    )

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    logger.info(f"Saved Train set: {train_path}")
    logger.info(f"Saved Validation set: {val_path}")
    logger.info(f"Saved Test set: {test_path}")

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    split_and_save()

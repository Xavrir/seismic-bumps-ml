"""Stratified 70/15/15 train/validation/test split for Seismic Bumps.

# TEMPORAL VALIDATION NOTE:
# True temporal cross-validation is not feasible with this dataset packaging
# because the ARFF file does not include explicit timestamps or shift identifiers
# that would allow strict time-ordered splitting.
# We use stratified random splitting as the best available alternative.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_arff import load_seismic_bumps, ARFF_PATH

RANDOM_STATE = 42
TARGET_COL = "class"


def make_splits(
    path: Path = ARFF_PATH,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load and split the Seismic Bumps dataset into stratified 70/15/15 splits.

    Split logic:
    - Step 1: Split into train_val (85%) and test (15%) with stratify=y
    - Step 2: Split train_val into train (70% of total) and validation (15% of total)
              using test_size = 15/85 ≈ 0.17647 with stratify=y_trainval

    All splits preserve the class distribution (stratified).

    Returns:
        Dict with keys 'train', 'validation', 'test'.
        Each value is a (X: DataFrame, y: Series) tuple.
        X has the target column removed.
        y is an integer Series (0=non-hazardous, 1=hazardous).
    """
    df = load_seismic_bumps(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Step 1: 85% train+val, 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state
    )

    # Step 2: 70% train, 15% val (of total) → test_size = 15/85
    val_frac = 15 / 85  # ≈ 0.17647
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac,
        stratify=y_trainval,
        random_state=random_state,
    )

    return {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }

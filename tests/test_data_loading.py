"""Smoke tests for the ARFF dataset loader."""

import pytest
import pandas as pd

from src.data.load_arff import load_seismic_bumps


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Load the dataset once for all tests in this module."""
    return load_seismic_bumps()


def test_shape(df: pd.DataFrame) -> None:
    """Dataset must have exactly 2584 rows and 19 columns."""
    assert df.shape == (2584, 19), f"Expected (2584, 19), got {df.shape}"


def test_target_dtype_is_int(df: pd.DataFrame) -> None:
    """Target column 'class' must be integer dtype."""
    assert pd.api.types.is_integer_dtype(df["class"]), (
        f"Expected integer dtype for 'class', got {df['class'].dtype}"
    )


def test_target_values(df: pd.DataFrame) -> None:
    """Target must contain only 0 (non-hazardous) and 1 (hazardous)."""
    assert set(df["class"].unique()) == {0, 1}, (
        f"Expected {{0, 1}}, got {set(df['class'].unique())}"
    )


def test_no_byte_strings_remain(df: pd.DataFrame) -> None:
    """No column should contain byte-string values after decoding."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            assert not isinstance(sample, bytes), (
                f"Column '{col}' still contains byte strings (e.g. {sample!r})"
            )

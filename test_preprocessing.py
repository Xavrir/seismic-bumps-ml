"""Smoke tests for preprocessing pipeline, splits, and metrics."""

import numpy as np
import pandas as pd
import pytest

from splits import make_splits
from preprocess import build_pipeline, ORDINAL_COLS, BINARY_COLS
from metrics import compute_metrics

TOTAL_ROWS = 2584


class TestSplitSizes:
    """Verify the stratified 70/15/15 split produces correct proportions."""

    @pytest.fixture(scope="class")
    def splits(self):
        return make_splits(random_state=42)

    def test_split_sizes(self, splits):
        """Split sizes should be approximately 70/15/15 (within ±2%)."""
        n_train = len(splits["train"][0])
        n_val = len(splits["validation"][0])
        n_test = len(splits["test"][0])

        assert n_train + n_val + n_test == TOTAL_ROWS, (
            f"Total rows mismatch: {n_train} + {n_val} + {n_test} != {TOTAL_ROWS}"
        )

        train_frac = n_train / TOTAL_ROWS
        val_frac = n_val / TOTAL_ROWS
        test_frac = n_test / TOTAL_ROWS

        assert abs(train_frac - 0.70) < 0.02, (
            f"Train fraction {train_frac:.3f} not ~0.70"
        )
        assert abs(val_frac - 0.15) < 0.02, f"Val fraction {val_frac:.3f} not ~0.15"
        assert abs(test_frac - 0.15) < 0.02, f"Test fraction {test_frac:.3f} not ~0.15"

    def test_split_stratification(self, splits):
        """Each split must contain positive-class samples (stratification works)."""
        for name in ("train", "validation", "test"):
            y = splits[name][1]
            pos_frac = y.mean()
            assert pos_frac > 0, f"Split '{name}' has no positive-class samples"
            # Class distribution should be roughly preserved (original ~6.6%)
            assert 0.03 < pos_frac < 0.12, (
                f"Split '{name}' positive fraction {pos_frac:.3f} looks off"
            )

    def test_split_keys(self, splits):
        """Splits dict must have exactly the expected keys."""
        assert set(splits.keys()) == {"train", "validation", "test"}

    def test_split_returns_dataframe_and_series(self, splits):
        """Each split value must be (DataFrame, Series)."""
        for name in ("train", "validation", "test"):
            X, y = splits[name]
            assert isinstance(X, pd.DataFrame), f"{name} X is not a DataFrame"
            assert isinstance(y, pd.Series), f"{name} y is not a Series"
            assert "class" not in X.columns, f"{name} X still contains 'class' column"


class TestPipeline:
    """Verify build_pipeline works for all model families."""

    @pytest.fixture
    def dummy_data(self):
        """Create a small dummy DataFrame mimicking the Seismic Bumps features."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "seismic": np.random.choice(["a", "b", "c", "d"], n),
                "seismoacoustic": np.random.choice(["a", "b", "c", "d"], n),
                "ghazard": np.random.choice(["a", "b", "c", "d"], n),
                "shift": np.random.choice(["N", "W"], n),
                "genergy": np.random.randn(n) * 100,
                "gpuls": np.random.randn(n) * 50,
                "gdenergy": np.random.randn(n) * 10,
                "gdpuls": np.random.randn(n) * 5,
                "energy": np.random.randn(n) * 1000,
                "maxenergy": np.random.randn(n) * 500,
                "nbumps": np.random.randint(0, 10, n),
                "nbumps2": np.random.randint(0, 5, n),
                "nbumps3": np.random.randint(0, 3, n),
                "nbumps4": np.random.randint(0, 2, n),
                "nbumps5": np.random.randint(0, 2, n),
                "nbumps6": np.random.randint(0, 1, n),
                "nbumps7": np.random.randint(0, 1, n),
                "nbumps89": np.random.randint(0, 1, n),
            }
        )
        return df

    def test_pipeline_logreg_fit_transform(self, dummy_data):
        """build_pipeline('logreg') can be fit and transform without error."""
        pipe = build_pipeline("logreg")
        result = pipe.fit_transform(dummy_data)
        assert result.shape[0] == len(dummy_data)
        # Should have: 3 ordinal + 1 binary + 14 numeric = 18 columns
        assert result.shape[1] == 18

    def test_pipeline_svm_fit_transform(self, dummy_data):
        """build_pipeline('svm') can be fit and transform without error."""
        pipe = build_pipeline("svm")
        result = pipe.fit_transform(dummy_data)
        assert result.shape[0] == len(dummy_data)

    def test_pipeline_rf_fit_transform(self, dummy_data):
        """build_pipeline('rf') can be fit and transform without error."""
        pipe = build_pipeline("rf")
        result = pipe.fit_transform(dummy_data)
        assert result.shape[0] == len(dummy_data)

    def test_pipeline_xgboost_fit_transform(self, dummy_data):
        """build_pipeline('xgboost') can be fit and transform without error."""
        pipe = build_pipeline("xgboost")
        result = pipe.fit_transform(dummy_data)
        assert result.shape[0] == len(dummy_data)

    def test_pipeline_invalid_model_type(self):
        """build_pipeline with unknown model_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_pipeline("neural_net")

    def test_logreg_scales_numerics(self, dummy_data):
        """LogReg pipeline should scale numeric columns (mean ≈ 0, std ≈ 1)."""
        pipe = build_pipeline("logreg")
        result = pipe.fit_transform(dummy_data)
        # Numeric columns start at index 4 (after 3 ordinal + 1 binary)
        numeric_part = result[:, 4:]
        col_means = np.abs(numeric_part.mean(axis=0))
        # After StandardScaler, means should be near 0
        assert np.all(col_means < 0.1), f"Numeric means not near 0: {col_means}"

    def test_rf_does_not_scale_numerics(self, dummy_data):
        """RF pipeline should NOT scale numeric columns (raw values preserved)."""
        pipe = build_pipeline("rf")
        result = pipe.fit_transform(dummy_data)
        # Numeric columns start at index 4
        numeric_part = result[:, 4:]
        # Raw values should have larger variance than scaled
        col_stds = numeric_part.std(axis=0)
        # At least some columns should have std >> 1 (raw numeric data)
        assert np.any(col_stds > 5), f"Numeric stds look scaled: {col_stds}"


class TestComputeMetrics:
    """Verify compute_metrics returns correct keys and values."""

    def test_compute_metrics_keys(self):
        """compute_metrics output must contain all required keys."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])

        result = compute_metrics(y_true, y_pred)
        expected_keys = {
            "hazardous_precision",
            "hazardous_recall",
            "hazardous_f1",
            "hazardous_f2",
            "roc_auc",
            "accuracy",
            "tn",
            "fp",
            "fn",
            "tp",
            "threshold",
        }
        assert set(result.keys()) == expected_keys

    def test_compute_metrics_values(self):
        """Spot-check metric values on a known example."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])
        # TP=2, FP=1, FN=1, TN=4
        result = compute_metrics(y_true, y_pred)
        assert result["tp"] == 2
        assert result["fp"] == 1
        assert result["fn"] == 1
        assert result["tn"] == 4
        assert result["accuracy"] == pytest.approx(6 / 8)
        assert result["roc_auc"] is None  # no y_score provided

    def test_compute_metrics_with_scores(self):
        """ROC-AUC should be computed when y_score is provided."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4])

        result = compute_metrics(y_true, y_pred, y_score=y_score)
        assert result["roc_auc"] is not None
        assert 0.0 <= result["roc_auc"] <= 1.0

"""Feature preprocessing and encoding pipelines for Seismic Bumps.

# LEAKAGE RULE: All preprocessing objects are fit on train split only.
# They are then applied (transform only) to validation and test splits.
# This pipeline is returned unfitted — caller fits it on train X only.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Categorical column definitions
ORDINAL_COLS = ["seismic", "seismoacoustic", "ghazard"]
BINARY_COLS = ["shift"]
TARGET_COL = "class"

# Category orderings — must match domain semantics
ORDINAL_CATEGORIES = [["a", "b", "c", "d"]] * 3  # same order for all 3 ordinal cols
BINARY_CATEGORIES = [["N", "W"]]  # N=0, W=1

# Model families that require numeric scaling
SCALED_MODELS = {"logreg", "svm"}
PASSTHROUGH_MODELS = {"rf", "xgboost"}
ALL_MODEL_TYPES = SCALED_MODELS | PASSTHROUGH_MODELS


def _get_numeric_cols(X_columns):
    """Return numeric column names (everything except ordinal, binary, and target)."""
    cat_cols = set(ORDINAL_COLS + BINARY_COLS + [TARGET_COL])
    return [c for c in X_columns if c not in cat_cols]


def build_pipeline(model_type: str, numeric_cols=None) -> Pipeline:
    """
    Build a preprocessing pipeline for the given model family.

    The pipeline uses ColumnTransformer to:
    - Ordinal-encode seismic/seismoacoustic/ghazard with category order [a, b, c, d]
    - Binary-encode shift with category order [N, W]
    - For logreg/svm: StandardScaler on numeric columns
    - For rf/xgboost: passthrough on numeric columns

    Args:
        model_type: One of 'logreg', 'svm', 'rf', 'xgboost'.
        numeric_cols: Explicit list of numeric column names. If None, a default
                      list is used that covers the standard Seismic Bumps features.

    Returns:
        An unfitted sklearn Pipeline with a single 'preprocessor' step.
    """
    if model_type not in ALL_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Must be one of {sorted(ALL_MODEL_TYPES)}"
        )

    if numeric_cols is None:
        # Default numeric columns from the Seismic Bumps dataset
        numeric_cols = [
            "genergy",
            "gpuls",
            "gdenergy",
            "gdpuls",
            "energy",
            "maxenergy",
            "nbumps",
            "nbumps2",
            "nbumps3",
            "nbumps4",
            "nbumps5",
            "nbumps6",
            "nbumps7",
            "nbumps89",
        ]

    # Ordinal encoding for ordered categorical features
    ordinal_transformer = OrdinalEncoder(
        categories=ORDINAL_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    # Binary encoding for shift column
    binary_transformer = OrdinalEncoder(
        categories=BINARY_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    # Numeric handling depends on model family
    if model_type in SCALED_MODELS:
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_transformer, ORDINAL_COLS),
            ("binary", binary_transformer, BINARY_COLS),
            ("numeric", numeric_transformer, numeric_cols),
        ],
        remainder="drop",  # drop any unexpected columns (e.g. target if present)
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])

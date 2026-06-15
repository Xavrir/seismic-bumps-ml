"""Load and clean the UCI Seismic Bumps ARFF dataset."""

from pathlib import Path
from scipy.io import arff
import pandas as pd

# Resolve the dataset relative to the repository root so the project runs anywhere
# (notebook, scripts, tests) without depending on a machine-specific download path.
ARFF_PATH = Path(__file__).resolve().parent / "data" / "raw" / "seismic-bumps.arff"


def load_seismic_bumps(path: Path = ARFF_PATH) -> pd.DataFrame:
    """
    Load seismic-bumps.arff and return a clean DataFrame.

    Transformations applied:
    - Decode all byte-string categorical columns to UTF-8 strings
    - Convert the 'class' target column to int (0 = non-hazardous, 1 = hazardous)

    Returns:
        DataFrame with shape (2584, 19), target dtype int, no byte strings.
    """
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    # Decode byte strings (ARFF returns categoricals as b'a' etc.)
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].str.decode("utf-8")
    # Convert target to int immediately — stable dtype for all downstream code
    df["class"] = df["class"].astype(int)
    return df

"""Data loading and cleaning for the Cleveland heart-disease dataset."""
from pathlib import Path

import pandas as pd

from src.config import CLEAN_PATH, COLUMNS, FEATURES, RAW_PATH, TARGET


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw UCI file (no header, '?' marks missing values)."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy.

    * all columns coerced to numeric (bad tokens become NaN)
    * exact duplicate rows dropped
    * target binarised: 0 = no disease, 1 = disease (original labels 1-4)

    Missing feature values are deliberately KEPT as NaN here. They are imputed
    inside the sklearn pipeline so that imputation statistics are learned from
    training folds only (no leakage).
    """
    out = df.copy()
    for col in COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.drop_duplicates().reset_index(drop=True)
    out = out.dropna(subset=[TARGET])
    out[TARGET] = (out[TARGET] > 0).astype(int)
    return out


def save_clean(df: pd.DataFrame, path: Path = CLEAN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    """Split a cleaned frame into features X and target y."""
    return df[FEATURES].copy(), df[TARGET].astype(int)

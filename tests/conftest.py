import numpy as np
import pandas as pd
import pytest

from src.config import COLUMNS


def make_synthetic(n=240, seed=0, missing=True) -> pd.DataFrame:
    """Synthetic patients with signal in a few features (no network needed)."""
    rng = np.random.default_rng(seed)
    risk = rng.random(n)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.normal(130, 15, n).round(),
        "chol": rng.normal(245, 45, n).round(),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": (170 - 40 * risk + rng.normal(0, 10, n)).round(),
        "exang": (risk + rng.normal(0, .3, n) > .6).astype(float),
        "oldpeak": np.clip(3 * risk + rng.normal(0, .5, n), 0, 6).round(1),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "target": (risk + rng.normal(0, .15, n) > .5).astype(int) * rng.choice([1, 2, 3], n),
    })
    if missing:
        df.loc[[3, 10], "ca"] = np.nan
        df.loc[[7], "thal"] = np.nan
    return df[COLUMNS]


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    from src.data import clean
    return clean(raw_df)


@pytest.fixture(scope="session")
def trained_model(clean_df):
    from sklearn.linear_model import LogisticRegression
    from src.data import split_xy
    from src.preprocessing import build_pipeline
    X, y = split_xy(clean_df)
    return build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y)


@pytest.fixture(scope="session")
def sample_patient():
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
            "ca": 0, "thal": 6}

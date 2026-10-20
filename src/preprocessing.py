"""Feature engineering + preprocessing pipeline (shared by training and serving)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, ENGINEERED_FEATURES, NUMERIC_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds ``hr_ratio`` = thalach / (220 - age).

    220 - age is the standard age-predicted maximum heart rate, so the ratio
    says how close the patient got to their theoretical maximum during the
    stress test (low values = poor chronotropic response).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        predicted_max = (220.0 - X["age"]).clip(lower=1.0)
        X["hr_ratio"] = X["thalach"] / predicted_max
        return X.replace([np.inf, -np.inf], np.nan)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Full model = feature engineering -> preprocessing -> classifier."""
    return Pipeline([
        ("fe", FeatureEngineer()),
        ("prep", build_preprocessor()),
        ("clf", estimator),
    ])

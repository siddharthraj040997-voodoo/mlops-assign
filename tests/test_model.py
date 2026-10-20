import joblib
import numpy as np
from sklearn.model_selection import cross_val_score

from src.data import split_xy


def test_proba_shape_and_range(trained_model, clean_df):
    X, _ = split_xy(clean_df)
    p = trained_model.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(p.sum(axis=1), 1)


def test_model_beats_chance_in_cv(trained_model, clean_df):
    X, y = split_xy(clean_df)
    auc = cross_val_score(trained_model, X, y, cv=3, scoring="roc_auc").mean()
    assert auc > 0.7


def test_pipeline_handles_missing_inputs(trained_model, clean_df):
    X, _ = split_xy(clean_df)
    X = X.head(5).copy()
    X.loc[X.index[0], ["ca", "thal", "chol"]] = np.nan
    assert trained_model.predict(X).shape == (5,)


def test_model_roundtrip_is_deterministic(trained_model, clean_df, tmp_path):
    X, _ = split_xy(clean_df)
    path = tmp_path / "m.joblib"
    joblib.dump(trained_model, path)
    reloaded = joblib.load(path)
    assert np.allclose(trained_model.predict_proba(X), reloaded.predict_proba(X))


def test_deployed_artifact_if_present():
    """If a trained artifact exists in models/, it must load and predict."""
    from src.config import MODEL_PATH
    import pytest
    if not MODEL_PATH.exists():
        pytest.skip("no trained artifact yet")
    from tests.conftest import make_synthetic
    from src.data import clean
    X, _ = split_xy(clean(make_synthetic()))
    assert joblib.load(MODEL_PATH).predict_proba(X.head(3)).shape == (3, 2)

import numpy as np
import pandas as pd

from src.data import split_xy
from src.preprocessing import FeatureEngineer, build_preprocessor


def test_feature_engineer_adds_hr_ratio():
    df = pd.DataFrame({"age": [60.0, 40.0], "thalach": [160.0, 180.0]})
    out = FeatureEngineer().transform(df)
    assert np.allclose(out["hr_ratio"], [1.0, 1.0])


def test_feature_engineer_does_not_mutate_input():
    df = pd.DataFrame({"age": [60.0], "thalach": [150.0]})
    FeatureEngineer().transform(df)
    assert "hr_ratio" not in df.columns


def test_feature_engineer_handles_extreme_age():
    df = pd.DataFrame({"age": [250.0], "thalach": [150.0]})
    assert np.isfinite(FeatureEngineer().transform(df)["hr_ratio"]).all()


def test_preprocessor_output_has_no_nans_and_is_scaled(clean_df):
    X, _ = split_xy(clean_df)
    Xe = FeatureEngineer().transform(X)
    prep = build_preprocessor()
    Z = prep.fit_transform(Xe)
    assert not np.isnan(Z).any()
    names = list(prep.get_feature_names_out())
    age_col = Z[:, names.index("age")]
    assert abs(age_col.mean()) < 1e-6 and abs(age_col.std() - 1) < 1e-6


def test_preprocessor_ignores_unseen_category(clean_df):
    X, _ = split_xy(clean_df)
    Xe = FeatureEngineer().transform(X)
    prep = build_preprocessor().fit(Xe)
    odd = Xe.head(2).copy()
    odd["thal"] = 99.0
    assert prep.transform(odd).shape[0] == 2

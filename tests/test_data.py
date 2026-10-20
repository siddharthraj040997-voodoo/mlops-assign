import numpy as np
import pandas as pd

from src.config import COLUMNS, FEATURES, TARGET
from src.data import clean, load_raw, split_xy


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}
    assert out[TARGET].sum() == (raw_df[TARGET] > 0).sum()


def test_clean_keeps_missing_for_pipeline_imputation(raw_df):
    out = clean(raw_df)
    assert out["ca"].isna().sum() == 2
    assert out["thal"].isna().sum() == 1


def test_clean_drops_duplicates():
    row = pd.DataFrame([[50, 1, 3, 120, 200, 0, 0, 150, 0, 1.0, 2, 0, 3, 1]],
                       columns=COLUMNS)
    out = clean(pd.concat([row, row], ignore_index=True))
    assert len(out) == 1


def test_clean_coerces_question_marks(tmp_path):
    p = tmp_path / "raw.data"
    p.write_text("63,1,1,145,233,1,2,150,0,2.3,3,?,6,0\n"
                 "67,1,4,160,286,0,2,108,1,1.5,2,3,3,2\n")
    df = clean(load_raw(p))
    assert np.isnan(df.loc[0, "ca"])
    assert df[TARGET].tolist() == [0, 1]
    assert df.dtypes.map(lambda t: np.issubdtype(t, np.number)).all()


def test_split_xy_shapes(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)

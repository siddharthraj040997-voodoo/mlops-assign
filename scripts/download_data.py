"""Download the UCI Heart Disease (Cleveland) data and build the cleaned CSV.

Usage:  python scripts/download_data.py

Sources are tried in order:
  1. UCI repository direct file URL
  2. ``ucimlrepo`` package (dataset id 45)
  3. GitHub mirror of the same UCI file (for networks that block UCI)
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLEAN_PATH, COLUMNS, RAW_PATH  # noqa: E402
from src.data import clean, load_raw, save_clean  # noqa: E402

UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
MIRROR_URL = ("https://raw.githubusercontent.com/constandinos/heart-disease/"
              "master/data/processed.cleveland.data")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def _validate(raw: bytes) -> bool:
    lines = [ln for ln in raw.decode().splitlines() if ln.strip()]
    return len(lines) == 303 and all(len(ln.split(",")) == 14 for ln in lines)


def _from_ucimlrepo() -> bytes:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=45)
    df = ds.data.features.copy()
    df["target"] = ds.data.targets.iloc[:, 0]
    return df[COLUMNS].to_csv(header=False, index=False, na_rep="?").encode()


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    attempts = [("UCI URL", lambda: _fetch(UCI_URL)),
                ("ucimlrepo", _from_ucimlrepo),
                ("GitHub mirror", lambda: _fetch(MIRROR_URL))]
    for name, getter in attempts:
        try:
            raw = getter()
            if not _validate(raw):
                raise ValueError("unexpected file shape")
            RAW_PATH.write_bytes(raw)
            print(f"[ok] downloaded raw data via {name} -> {RAW_PATH}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {name} failed: {exc}")
    else:
        sys.exit("All download sources failed. Place processed.cleveland.data "
                 f"manually at {RAW_PATH}")

    df = clean(load_raw())
    save_clean(df)
    print(f"[ok] cleaned dataset {df.shape} -> {CLEAN_PATH}")
    print(f"     missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")
    print(f"     class balance: {df['target'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

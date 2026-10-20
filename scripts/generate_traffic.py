"""Send realistic traffic to the API so Prometheus/Grafana have data to show.

Samples real rows from the cleaned dataset (plus a few invalid requests).
Usage: python scripts/generate_traffic.py --url http://localhost:8000 --n 300
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FEATURES  # noqa: E402
from src.data import load_clean  # noqa: E402

INT_COLS = {"age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}


def post(url, payload):
    req = urllib.request.Request(url + "/predict", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as e:
        return e.code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--delay", type=float, default=0.05)
    a = ap.parse_args()
    df = load_clean().dropna(subset=FEATURES)
    codes = {}
    for _ in range(a.n):
        row = df.sample(1).iloc[0][FEATURES].to_dict()
        payload = {k: (int(v) if k in INT_COLS else float(v)) for k, v in row.items()}
        if random.random() < 0.05:
            payload["age"] = -1  # deliberately invalid -> 422
        c = post(a.url, payload)
        codes[c] = codes.get(c, 0) + 1
        time.sleep(a.delay)
    print("status codes:", codes)


if __name__ == "__main__":
    main()

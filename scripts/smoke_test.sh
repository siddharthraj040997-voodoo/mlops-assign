#!/usr/bin/env bash
# Usage: bash scripts/smoke_test.sh [base_url]   (default http://localhost:8000)
set -euo pipefail
URL="${1:-http://localhost:8000}"
echo "== /health";  curl -fsS "$URL/health"; echo
echo "== /predict (high-risk profile)"
curl -fsS -X POST "$URL/predict" -H 'Content-Type: application/json' -d '{
  "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
  "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2,
  "ca": 3, "thal": 3}'; echo
echo "== /predict (low-risk profile)"
curl -fsS -X POST "$URL/predict" -H 'Content-Type: application/json' -d '{
  "age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0,
  "restecg": 2, "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 1,
  "ca": 0, "thal": 3}'; echo
echo "== invalid input must be rejected with 422"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$URL/predict" \
  -H 'Content-Type: application/json' -d '{"age": -5}')
[ "$code" = "422" ] && echo "got 422 OK" || { echo "expected 422, got $code"; exit 1; }

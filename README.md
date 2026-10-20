# Heart Disease Risk Prediction - End-to-End MLOps

Predicts presence of heart disease from 13 clinical features (UCI Cleveland, 303 patients) and ships the model as a
tested, containerised, Kubernetes-deployable, monitored API.

- **Student:** `Siddharth Raj`
- **BITS ID:** `2025ae05706` (`2025ae05706@wilp.bits-pilani.ac.in`)
- **Course:** Machine Learning Operations (MLOps) AIMLCZG523 - Assignment 01
- **GitHub Repository:** [`https://github.com/siddharthraj040997-voodoo/mlops-assign`](https://github.com/siddharthraj040997-voodoo/mlops-assign)
- **Full Report:** [`report/REPORT.pdf`](report/REPORT.pdf) / [`report/REPORT.html`](report/REPORT.html)
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Monitoring Stack:** Prometheus (`:9090`) & Grafana (`:3000`, `admin`/`admin`)
- **Demo Video Script:** [`DEMO_VIDEO_SCRIPT.txt`](DEMO_VIDEO_SCRIPT.txt)


| | |
|---|---|
| Model | Logistic Regression in an sklearn `Pipeline` (feature engineering + impute/scale/one-hot) |
| CV ROC-AUC (5-fold) | 0.903 |
| Hold-out | acc 0.885, precision 0.839, recall 0.929, ROC-AUC 0.966 |
| Tracking | MLflow (32 runs: 3 model families, 28 tuning candidates, 1 final) |
| Serving | FastAPI `/predict`, `/health`, `/metrics` in a `python:3.12-slim` image |
| Ops | GitHub Actions, Kubernetes (Deployment, LoadBalancer Service, HPA), Prometheus + Grafana |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/download_data.py      # data/raw + data/heart_clean.csv
python -m src.eda                    # figures -> reports/figures
python -m src.train                  # MLflow runs + models/heart_pipeline.joblib
pytest && flake8 .
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Run the API
```bash
# local
uvicorn api.app:app --port 8000          # docs at http://localhost:8000/docs

# docker
docker build -t heart-api:1.0.0 . && docker run -p 8000:8000 heart-api:1.0.0
bash scripts/smoke_test.sh

# full local monitoring stack (API :8000, Prometheus :9090, Grafana :3000 admin/admin)
docker compose up --build
python scripts/generate_traffic.py --n 300

# Kubernetes (Minikube or Docker Desktop)
bash scripts/deploy_k8s.sh
```

Example request:
```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
 "age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
 "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
# {"prediction":0,"label":"no_disease","confidence":0.8504,"probability_disease":0.1496,"model_version":"1.0.0"}
```

## Layout
```
src/            config, data cleaning, preprocessing pipeline, EDA, training (MLflow)
api/app.py      FastAPI service + Prometheus metrics + JSON request logging
tests/          pytest (data, preprocessing, model, API)
notebooks/      01_eda, 02_training, 03_inference (executed)
scripts/        download_data, deploy_k8s, smoke_test, generate_traffic, build_report ...
k8s/            namespace, deployment, service, hpa, ingress, monitoring/
monitoring/     prometheus.yml, Grafana provisioning + dashboard JSON
.github/workflows/ci.yml
models/         heart_pipeline.joblib, metrics.json
screenshots/    evidence for the report (replace the placeholders)
report/         REPORT.md / REPORT.pdf, architecture.png
```
Dataset: Detrano et al., UCI Machine Learning Repository, Heart Disease (id 45). For education only - not a medical device.

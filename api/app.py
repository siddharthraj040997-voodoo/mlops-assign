"""FastAPI serving layer for the heart-disease classifier.

Endpoints
  POST /predict   JSON patient record -> prediction + confidence
  GET  /health    liveness/readiness
  GET  /metrics   Prometheus metrics
"""
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from src.config import FEATURES  # noqa: F401  (also guarantees src is importable)

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# ---------------------------------------------------------------- logging
logger = logging.getLogger("heart-api")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    log_file = os.getenv("LOG_FILE")
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
logger.setLevel(logging.INFO)
logger.propagate = False


def log_event(**fields):
    fields["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    logger.info(json.dumps(fields, default=str))


# ---------------------------------------------------------------- metrics
REQUESTS = Counter("heart_api_requests_total", "HTTP requests",
                   ["method", "path", "status"])
LATENCY = Histogram("heart_api_request_latency_seconds", "Request latency",
                    ["path"],
                    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions by class",
                      ["label"])
CONFIDENCE = Histogram("heart_api_prediction_confidence",
                       "Confidence of returned prediction",
                       buckets=(.5, .6, .7, .8, .9, .95, 1.0))
ERRORS = Counter("heart_api_errors_total", "Unhandled errors")
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if the model is loaded")


# ---------------------------------------------------------------- schema
class PatientFeatures(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type")
    trestbps: float = Field(..., ge=50, le=300, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=30, le=250, description="Max heart rate")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: Literal[1, 2, 3] = Field(..., description="ST slope")
    ca: Literal[0, 1, 2, 3] = Field(..., description="Major vessels coloured")
    thal: Literal[3, 6, 7] = Field(..., description="Thalassemia code")

    model_config = {"json_schema_extra": {"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}}}


class Prediction(BaseModel):
    prediction: int = Field(..., description="1 = heart disease, 0 = none")
    label: str
    confidence: float = Field(..., description="Probability of predicted class")
    probability_disease: float
    model_version: str


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", "models/heart_pipeline.joblib")
    try:
        app.state.model = joblib.load(path)
        MODEL_LOADED.set(1)
        log_event(event="model_loaded", path=path, version=MODEL_VERSION)
    except Exception as exc:  # noqa: BLE001
        app.state.model = None
        MODEL_LOADED.set(0)
        log_event(event="model_load_failed", path=path, error=str(exc))
    yield


app = FastAPI(title="Heart Disease Risk API", version=MODEL_VERSION,
              description="Predicts presence of heart disease from clinical data.",
              lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    except Exception:
        ERRORS.inc()
        raise
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(elapsed)
            log_event(event="request", request_id=rid, method=request.method,
                      path=path, status=status,
                      latency_ms=round(elapsed * 1000, 2),
                      client=request.client.host if request.client else None)


@app.get("/")
def root():
    return {"service": "heart-disease-api", "docs": "/docs",
            "endpoints": ["/predict", "/health", "/metrics"]}


@app.get("/health")
def health(request: Request):
    ok = request.app.state.model is not None
    if not ok:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientFeatures, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    frame = pd.DataFrame([patient.model_dump()])[FEATURES]
    p_disease = float(model.predict_proba(frame)[0, 1])
    pred = int(p_disease >= 0.5)
    confidence = p_disease if pred == 1 else 1.0 - p_disease
    label = "disease" if pred else "no_disease"
    PREDICTIONS.labels(label).inc()
    CONFIDENCE.observe(confidence)
    log_event(event="prediction", prediction=pred,
              confidence=round(confidence, 4), age=patient.age)
    return Prediction(prediction=pred, label=label,
                      confidence=round(confidence, 4),
                      probability_disease=round(p_disease, 4),
                      model_version=MODEL_VERSION)

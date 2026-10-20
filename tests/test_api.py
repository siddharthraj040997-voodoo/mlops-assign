import joblib
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained_model, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_contract(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0 <= body["probability_disease"] <= 1
    assert "x-request-id" in r.headers


@pytest.mark.parametrize("field,value", [
    ("age", 0), ("sex", 2), ("cp", 9), ("thal", 5), ("oldpeak", -1), ("chol", "abc")])
def test_predict_rejects_invalid(client, sample_patient, field, value):
    bad = {**sample_patient, field: value}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_missing_field(client, sample_patient):
    bad = dict(sample_patient)
    bad.pop("age")
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_exposed_after_traffic(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text
    assert "heart_api_model_loaded 1.0" in text


def test_health_503_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    from api.app import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503

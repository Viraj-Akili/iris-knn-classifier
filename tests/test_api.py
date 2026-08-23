"""
Comprehensive integration tests for FastAPI ML inference service.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app, metrics_tracker, model_service


@pytest.fixture(autouse=True)
def client():
    """Test client fixture with lifespan context."""
    metrics_tracker.reset()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_healthy(client: TestClient):
    """Verify /health reports healthy status and model loaded state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"
    assert data["service"] == "iris-ml-api"
    assert "X-Request-ID" in response.headers


def test_model_info_endpoint(client: TestClient):
    """Verify /model returns full champion model metadata and benchmark scores."""
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Support Vector Machine"
    assert data["model_version"] == "1.0.0"
    assert data["model_type"] == "SVC"
    assert len(data["feature_names"]) == 4
    assert set(data["class_names"]) == {"setosa", "versicolor", "virginica"}
    assert 0.0 <= data["cv_accuracy"] <= 1.0
    assert 0.0 <= data["test_accuracy"] <= 1.0
    assert "best_hyperparameters" in data


def test_predict_valid_sample(client: TestClient):
    """Verify /predict returns valid class, confidence, probabilities, and latency."""
    payload = {
        "sepal_length": 5.8,
        "sepal_width": 2.7,
        "petal_length": 4.1,
        "petal_width": 1.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ["setosa", "versicolor", "virginica"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 3
    for cls_name in ["setosa", "versicolor", "virginica"]:
        assert cls_name in data["probabilities"]
        assert 0.0 <= data["probabilities"][cls_name] <= 1.0

    # Probabilities sum to approximately 1.0
    total_prob = sum(data["probabilities"].values())
    assert abs(total_prob - 1.0) < 0.01

    assert data["inference_latency_ms"] >= 0.0
    assert len(data["request_id"]) > 0
    assert data["model_version"] == "1.0.0"
    assert data["model_name"] == "Support Vector Machine"
    assert response.headers.get("X-Request-ID") == data["request_id"]


def test_predict_distinct_flower_species(client: TestClient):
    """Verify predictions on typical points for setosa and virginica."""
    # Typical Setosa features
    setosa_payload = {
        "sepal_length": 5.0,
        "sepal_width": 3.6,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    res_setosa = client.post("/predict", json=setosa_payload)
    assert res_setosa.status_code == 200
    assert res_setosa.json()["prediction"] == "setosa"
    assert res_setosa.json()["confidence"] > 0.80

    # Typical Virginica features
    virginica_payload = {
        "sepal_length": 6.5,
        "sepal_width": 3.0,
        "petal_length": 5.5,
        "petal_width": 2.0,
    }
    res_virginica = client.post("/predict", json=virginica_payload)
    assert res_virginica.status_code == 200
    assert res_virginica.json()["prediction"] == "virginica"
    assert res_virginica.json()["confidence"] > 0.80


def test_predict_missing_fields_returns_422(client: TestClient):
    """Verify missing required fields triggers HTTP 422 with structured error."""
    payload = {
        "sepal_length": 5.8,
        "sepal_width": 2.7,
        # missing petal_length and petal_width
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["error_code"] == "VALIDATION_ERROR"
    assert "petal_length" in data["detail"]


def test_predict_invalid_types_returns_422(client: TestClient):
    """Verify non-numeric types trigger HTTP 422."""
    payload = {
        "sepal_length": "five_point_eight",
        "sepal_width": 2.7,
        "petal_length": 4.1,
        "petal_width": 1.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["error_code"] == "VALIDATION_ERROR"


def test_predict_nan_inf_returns_422(client: TestClient):
    """Verify NaN and Infinity strings/values are rejected with HTTP 422."""
    # Raw JSON string containing NaN
    raw_nan_json = '{"sepal_length": NaN, "sepal_width": 2.7, "petal_length": 4.1, "petal_width": 1.0}'
    res_nan = client.post("/predict", content=raw_nan_json, headers={"Content-Type": "application/json"})
    assert res_nan.status_code == 422

    # Raw JSON string containing Infinity
    raw_inf_json = '{"sepal_length": Infinity, "sepal_width": 2.7, "petal_length": 4.1, "petal_width": 1.0}'
    res_inf = client.post("/predict", content=raw_inf_json, headers={"Content-Type": "application/json"})
    assert res_inf.status_code == 422

    # String representations of NaN/Infinity in JSON
    str_nan_payload = {
        "sepal_length": "NaN",
        "sepal_width": 2.7,
        "petal_length": 4.1,
        "petal_width": 1.0,
    }
    res_str_nan = client.post("/predict", json=str_nan_payload)
    assert res_str_nan.status_code == 422


def test_predict_out_of_bounds_returns_422(client: TestClient):
    """Verify non-sensical or negative biological measurements are rejected."""
    # Negative value
    neg_payload = {
        "sepal_length": -5.0,
        "sepal_width": 2.7,
        "petal_length": 4.1,
        "petal_width": 1.0,
    }
    res_neg = client.post("/predict", json=neg_payload)
    assert res_neg.status_code == 422

    # Absurdly large value (> 15 cm for an iris flower petal)
    large_payload = {
        "sepal_length": 5.8,
        "sepal_width": 2.7,
        "petal_length": 99.0,
        "petal_width": 1.0,
    }
    res_large = client.post("/predict", json=large_payload)
    assert res_large.status_code == 422


def test_runtime_metrics_tracking(client: TestClient):
    """Verify /metrics tracks requests, successful inferences, and failures."""
    # Make 2 valid predictions
    payload = {
        "sepal_length": 5.8,
        "sepal_width": 2.7,
        "petal_length": 4.1,
        "petal_width": 1.0,
    }
    client.post("/predict", json=payload)
    client.post("/predict", json=payload)

    # Make 1 invalid prediction
    client.post("/predict", json={"sepal_length": -1.0})

    # Query metrics via JSON Accept header
    res = client.get("/metrics", headers={"Accept": "application/json"})
    assert res.status_code == 200
    metrics = res.json()
    assert metrics["successful_predictions"] == 2
    assert metrics["failed_requests"] >= 1
    assert metrics["total_requests"] >= 3
    assert metrics["avg_latency_ms"] >= 0.0
    assert metrics["uptime_seconds"] >= 0.0
    assert "note" in metrics


def test_model_unavailable_returns_503(client: TestClient, monkeypatch):
    """Verify 503 error is cleanly returned when model service is uninitialized."""
    # Temporarily mark model_service as unloaded
    monkeypatch.setattr(model_service, "_is_loaded", False)

    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "degraded"
    assert health_res.json()["model_loaded"] is False

    model_res = client.get("/model")
    assert model_res.status_code == 503

    predict_res = client.post(
        "/predict",
        json={
            "sepal_length": 5.8,
            "sepal_width": 2.7,
            "petal_length": 4.1,
            "petal_width": 1.0,
        },
    )
    assert predict_res.status_code == 503
    assert predict_res.json()["error_code"] == "HTTP_503"


def test_model_service_load_and_predict_exceptions(tmp_path: Path):
    """Verify ModelService raises expected domain exceptions on missing files or invalid states."""
    from app.model_service import ModelInferenceError, ModelLoadError, ModelService

    svc = ModelService()

    # Predict before loading
    with pytest.raises(ModelInferenceError):
        svc.predict({"sepal_length": 5.0, "sepal_width": 3.0, "petal_length": 1.0, "petal_width": 0.2})

    # Get metadata before loading
    with pytest.raises(ModelLoadError):
        svc.get_metadata()

    # Load non-existent file
    with pytest.raises(ModelLoadError):
        svc.load(tmp_path / "missing.joblib", tmp_path / "missing.json")


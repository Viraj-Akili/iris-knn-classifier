"""
Integration and unit tests for ML Observability, Prometheus metrics,
health/readiness probes, and data drift detection.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, metrics_tracker, model_service, observability_tracker
from src.monitoring.drift import (
    DataDriftDetector,
    calculate_ks_test,
    calculate_psi,
    calculate_wasserstein,
)


@pytest.fixture(autouse=True)
def client():
    """Test client fixture with clean metrics state and lifespan."""
    metrics_tracker.reset()
    observability_tracker.reset()
    with TestClient(app) as test_client:
        yield test_client


def test_prometheus_metrics_text_format(client: TestClient):
    """Verify /metrics returns standard Prometheus text exposition format."""
    # Send a prediction to trigger metric recording
    payload = {
        "sepal_length": 5.8,
        "sepal_width": 2.7,
        "petal_length": 4.1,
        "petal_width": 1.0,
    }
    client.post("/predict", json=payload)

    # Request Prometheus metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    text = response.text

    assert "iris_prediction_requests_total" in text
    assert "iris_prediction_success_total" in text
    assert "iris_predictions_by_class_total" in text
    assert "iris_prediction_latency_seconds_bucket" in text
    assert "iris_model_info" in text


def test_prometheus_metrics_json_accept_header(client: TestClient):
    """Verify /metrics returns JSON when application/json Accept header is sent."""
    payload = {
        "sepal_length": 5.8,
        "sepal_width": 2.7,
        "petal_length": 4.1,
        "petal_width": 1.0,
    }
    client.post("/predict", json=payload)

    response = client.get("/metrics", headers={"Accept": "application/json"})
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    data = response.json()
    assert "total_requests" in data
    assert "successful_predictions" in data


def test_health_and_readiness_probes(client: TestClient):
    """Verify distinct semantics of /health (liveness) vs /readiness."""
    # 1. Health (Liveness)
    health_res = client.get("/health")
    assert health_res.status_code == 200
    health_data = health_res.json()
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] is True

    # 2. Readiness
    readiness_res = client.get("/readiness")
    assert readiness_res.status_code == 200
    ready_data = readiness_res.json()
    assert ready_data["status"] == "ready"
    assert ready_data["model_loaded"] is True
    assert ready_data["model_name"] == "Support Vector Machine"
    assert ready_data["checks"]["model_artifact"] == "loaded"
    assert ready_data["checks"]["inference_engine"] == "ready"


def test_readiness_when_model_unloaded(client: TestClient, monkeypatch):
    """Verify /readiness returns 503 while /health returns 200 when model is offline."""
    monkeypatch.setattr(model_service, "_is_loaded", False)

    # Liveness probe still passes (process is alive)
    health_res = client.get("/health")
    assert health_res.status_code == 200

    # Readiness probe fails (cannot accept prediction traffic)
    readiness_res = client.get("/readiness")
    assert readiness_res.status_code == 503
    ready_data = readiness_res.json()
    assert ready_data["status"] == "not_ready"
    assert ready_data["model_loaded"] is False
    assert ready_data["checks"]["model_artifact"] == "not_loaded"


def test_observability_summary_endpoint(client: TestClient):
    """Verify /observability/summary aggregates class distributions, latency, and features."""
    # Send setosa and virginica samples
    samples = [
        {"sepal_length": 5.0, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 2.0},
        {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 4.1, "petal_width": 1.0},
    ]
    for s in samples:
        client.post("/predict", json=s)

    response = client.get("/observability/summary")
    assert response.status_code == 200
    data = response.json()

    assert data["service"] == "iris-ml-api"
    assert data["model_name"] == "Support Vector Machine"
    assert data["total_requests"] >= 3
    assert data["successful_predictions"] >= 3

    # Prediction class distribution
    assert isinstance(data["predictions_by_class"], dict)
    assert sum(data["predictions_by_class"].values()) >= 3

    # Confidence tier distribution
    assert "high" in data["confidence_distribution"]
    assert "medium" in data["confidence_distribution"]
    assert "low" in data["confidence_distribution"]

    # Latency percentiles
    lat = data["latency_statistics_ms"]
    assert lat["mean_ms"] >= 0.0
    assert lat["p50_ms"] >= 0.0
    assert lat["p95_ms"] >= 0.0
    assert lat["p99_ms"] >= 0.0

    # Feature aggregates
    feats = data["feature_aggregates"]
    for feat_key in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        assert feat_key in feats
        assert feats[feat_key]["count"] >= 3
        assert feats[feat_key]["mean"] > 0.0
        assert feats[feat_key]["min"] > 0.0

    # Health state
    assert data["health_state"]["liveness"] == "healthy"
    assert data["health_state"]["readiness"] == "ready"


def test_structured_prediction_jsonl_logging(tmp_path: Path):
    """Verify prediction events are appended to JSONL log file correctly."""
    log_file = tmp_path / "test_predictions.jsonl"
    custom_tracker = observability_tracker
    custom_tracker.prediction_log_path = log_file
    custom_tracker.enable_prediction_logging = True

    custom_tracker.record_prediction(
        request_id="test-uuid-12345",
        features={"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 4.1, "petal_width": 1.0},
        prediction="versicolor",
        confidence=0.9608,
        latency_ms=1.25,
        model_name="Support Vector Machine",
        model_version="1.0.0",
    )

    assert log_file.exists()
    with open(log_file, encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["request_id"] == "test-uuid-12345"
        assert entry["prediction"] == "versicolor"
        assert entry["confidence"] == 0.9608
        assert entry["model_version"] == "1.0.0"


def test_statistical_drift_detector_functions():
    """Verify KS-test, Wasserstein distance, and PSI computations."""
    np.random.seed(42)
    dist_a = np.random.normal(5.0, 1.0, 500)
    dist_b = np.random.normal(5.0, 1.0, 500)  # Same distribution
    dist_c = np.random.normal(8.0, 1.5, 500)  # Shifted distribution

    # Identical distributions
    ks_stat_same, p_val_same = calculate_ks_test(dist_a, dist_b)
    assert p_val_same > 0.05  # Fail to reject null hypothesis (no drift)
    wass_same = calculate_wasserstein(dist_a, dist_b)
    assert wass_same < 0.2
    psi_same = calculate_psi(dist_a, dist_b)
    assert psi_same < 0.10

    # Self-comparison (identical)
    assert calculate_psi(dist_a, dist_a) == pytest.approx(0.0, abs=1e-5)

    # Shifted distributions
    ks_stat_diff, p_val_diff = calculate_ks_test(dist_a, dist_c)
    assert p_val_diff < 0.01  # Statistically significant difference
    wass_diff = calculate_wasserstein(dist_a, dist_c)
    assert wass_diff > 2.0
    psi_diff = calculate_psi(dist_a, dist_c)
    assert psi_diff >= 0.20


def test_data_drift_detector_class():
    """Verify DataDriftDetector evaluates dataset shifts into tabular reports."""
    np.random.seed(42)
    base_df = pd.DataFrame({
        "sepal_length": np.random.normal(5.8, 0.8, 100),
        "petal_length": np.random.normal(3.8, 1.7, 100),
    })

    # Shifted test batch
    curr_df = pd.DataFrame({
        "sepal_length": np.random.normal(7.5, 0.8, 100),  # shifted
        "petal_length": np.random.normal(3.8, 1.7, 100),  # stable
    })

    detector = DataDriftDetector(baseline_df=base_df, ks_alpha=0.05, psi_drift_threshold=0.20)
    summary = detector.evaluate_drift(curr_df)

    assert summary.total_features_evaluated == 2
    assert summary.drifted_features_count >= 1
    assert summary.dataset_drift_detected is True

    table_df = summary.to_dataframe()
    assert len(table_df) == 2
    assert "Feature" in table_df.columns
    assert "KS p-val" in table_df.columns
    assert "Status" in table_df.columns

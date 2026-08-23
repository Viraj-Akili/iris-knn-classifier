"""
Unit tests for the Streamlit frontend API client.
Tests communication with FastAPI backend, payload serialization, and exception mapping.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from frontend.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiUnavailableError,
    ApiValidationError,
    IrisApiClient,
)


@pytest.fixture
def api_client():
    return IrisApiClient(base_url="http://mock-fastapi:8000")


def test_api_client_check_connection_true(api_client):
    """Verify check_connection returns True on 200 response."""
    with patch("httpx.Client.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        assert api_client.check_connection() is True


def test_api_client_check_connection_false(api_client):
    """Verify check_connection returns False when connection fails."""
    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
        assert api_client.check_connection() is False


def test_api_client_get_health_success(api_client):
    """Verify get_health returns parsed JSON."""
    mock_payload = {"status": "healthy", "model_loaded": True, "service": "iris-ml-api"}
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = mock_payload
        mock_get.return_value = mock_resp

        data = api_client.get_health()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


def test_api_client_get_readiness_success(api_client):
    """Verify get_readiness returns readiness state."""
    mock_payload = {"status": "ready", "model_loaded": True, "model_name": "Support Vector Machine"}
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = mock_payload
        mock_get.return_value = mock_resp

        data = api_client.get_readiness()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True


def test_api_client_get_model_info_success(api_client):
    """Verify get_model_info returns model metadata."""
    mock_payload = {
        "model_name": "Support Vector Machine",
        "model_version": "1.0.0",
        "cv_accuracy": 0.975,
        "test_accuracy": 0.9333,
    }
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = mock_payload
        mock_get.return_value = mock_resp

        data = api_client.get_model_info()
        assert data["model_name"] == "Support Vector Machine"
        assert data["cv_accuracy"] == 0.975


def test_api_client_predict_success(api_client):
    """Verify predict sends measurements and parses response."""
    mock_payload = {
        "request_id": "test-req-123",
        "prediction": "versicolor",
        "confidence": 0.9608,
        "probabilities": {"setosa": 0.0294, "versicolor": 0.9608, "virginica": 0.0098},
        "model_name": "Support Vector Machine",
        "model_version": "1.0.0",
        "inference_latency_ms": 0.742,
    }
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = mock_payload
        mock_post.return_value = mock_resp

        res = api_client.predict(5.8, 2.7, 4.1, 1.0)
        assert res["prediction"] == "versicolor"
        assert res["confidence"] == 0.9608
        assert res["probabilities"]["versicolor"] == 0.9608


def test_api_client_predict_validation_error_422(api_client):
    """Verify HTTP 422 triggers ApiValidationError."""
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock(status_code=422)
        mock_resp.json.return_value = {"detail": "sepal_length: Input should be greater than or equal to 0.1"}
        mock_post.return_value = mock_resp

        with pytest.raises(ApiValidationError) as exc:
            api_client.predict(-1.0, 2.7, 4.1, 1.0)
        assert "Validation failed" in str(exc.value) or "sepal_length" in str(exc.value)


def test_api_client_predict_unavailable_503(api_client):
    """Verify HTTP 503 triggers ApiUnavailableError."""
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock(status_code=503)
        mock_post.return_value = mock_resp

        with pytest.raises(ApiUnavailableError):
            api_client.predict(5.8, 2.7, 4.1, 1.0)


def test_api_client_connection_error(api_client):
    """Verify connection failure triggers ApiConnectionError."""
    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(ApiConnectionError):
            api_client.get_health()


def test_api_client_observability_summary(api_client):
    """Verify get_observability_summary returns summary payload."""
    mock_payload = {
        "service": "iris-ml-api",
        "total_requests": 150,
        "successful_predictions": 150,
        "failed_requests": 0,
    }
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = mock_payload
        mock_get.return_value = mock_resp

        res = api_client.get_observability_summary()
        assert res["service"] == "iris-ml-api"
        assert res["total_requests"] == 150


def test_api_client_prometheus_metrics(api_client):
    """Verify get_prometheus_metrics returns text."""
    mock_text = "# HELP iris_prediction_requests_total\niris_prediction_requests_total 10\n"
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock(status_code=200, text=mock_text)
        mock_get.return_value = mock_resp

        res = api_client.get_prometheus_metrics()
        assert "iris_prediction_requests_total" in res


def test_api_client_status_error_branches(api_client):
    """Verify general status errors trigger ApiClientError."""
    with patch("httpx.Client.get", side_effect=httpx.HTTPStatusError("500 Server Error", request=MagicMock(), response=MagicMock(status_code=500))):
        with pytest.raises(ApiClientError):
            api_client.get_health()

    with patch("httpx.Client.get", side_effect=Exception("Timeout")):
        with pytest.raises(ApiClientError):
            api_client.get_readiness()

    with patch("httpx.Client.get", side_effect=Exception("Observability error")):
        with pytest.raises(ApiClientError):
            api_client.get_observability_summary()

    with patch("httpx.Client.get", side_effect=Exception("Metrics error")):
        with pytest.raises(ApiClientError):
            api_client.get_prometheus_metrics()


def test_app_config_from_env(monkeypatch):
    """Verify AppConfig parses environment overrides."""
    from app.config import AppConfig

    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "9090")
    monkeypatch.setenv("API_DEBUG", "true")
    monkeypatch.setenv("CORS_ORIGINS", "http://app.io, http://admin.io")
    monkeypatch.setenv("HIGH_CONFIDENCE_THRESHOLD", "0.95")
    monkeypatch.setenv("LOW_CONFIDENCE_THRESHOLD", "0.65")
    monkeypatch.setenv("ENABLE_PREDICTION_LOGGING", "false")

    cfg = AppConfig.from_env()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9090
    assert cfg.debug is True
    assert "http://app.io" in cfg.allowed_origins
    assert cfg.high_confidence_threshold == 0.95
    assert cfg.low_confidence_threshold == 0.65
    assert cfg.enable_prediction_logging is False


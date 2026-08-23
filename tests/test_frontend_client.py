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


def test_altair_charts_schema_validity():
    """Verify all frontend Altair charts serialize cleanly to valid Vega-Lite specs without schema errors."""
    import altair as alt
    import pandas as pd

    from frontend.components.charts import (
        render_class_distribution_chart,
        render_confidence_chart,
        render_feature_distribution_chart,
        render_probability_chart,
    )

    with patch("streamlit.altair_chart") as mock_st_chart, patch("streamlit.info"):
        # 1. Probability chart
        render_probability_chart({"setosa": 0.1, "versicolor": 0.8, "virginica": 0.1})
        assert mock_st_chart.called
        chart1 = mock_st_chart.call_args[0][0]
        assert chart1.to_dict() is not None

        # 2. Prediction class distribution chart
        mock_st_chart.reset_mock()
        render_class_distribution_chart({"setosa": 10, "versicolor": 25, "virginica": 15})
        assert mock_st_chart.called
        chart2 = mock_st_chart.call_args[0][0]
        assert chart2.to_dict() is not None

        # 3. Confidence distribution chart
        mock_st_chart.reset_mock()
        render_confidence_chart({"high": 30, "medium": 15, "low": 5})
        assert mock_st_chart.called
        chart3 = mock_st_chart.call_args[0][0]
        assert chart3.to_dict() is not None

        # 4. Feature distribution chart
        mock_st_chart.reset_mock()
        feature_stats = {
            "sepal_length": {"count": 10, "mean": 5.8, "std": 0.8, "min": 4.3, "max": 7.9},
            "sepal_width": {"count": 10, "mean": 3.0, "std": 0.4, "min": 2.0, "max": 4.4},
            "petal_length": {"count": 10, "mean": 3.7, "std": 1.7, "min": 1.0, "max": 6.9},
            "petal_width": {"count": 10, "mean": 1.2, "std": 0.7, "min": 0.1, "max": 2.5},
        }
        render_feature_distribution_chart(feature_stats)
        assert mock_st_chart.called
        chart4 = mock_st_chart.call_args[0][0]
        assert chart4.to_dict() is not None

    # 5. Explainability weights chart (verifying valid scheme='tealblues')
    weights_data = [
        {"Feature": "Petal Length (cm)", "Importance Score": 0.468, "Impact Level": "Primary Separator"},
        {"Feature": "Petal Width (cm)", "Importance Score": 0.382, "Impact Level": "Primary Separator"},
        {"Feature": "Sepal Width (cm)", "Importance Score": 0.096, "Impact Level": "Secondary Boundary Stabilizer"},
        {"Feature": "Sepal Length (cm)", "Importance Score": 0.054, "Impact Level": "Minor Discriminator"},
    ]
    df_weights = pd.DataFrame(weights_data)
    chart5 = (
        alt.Chart(df_weights)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, height=36)
        .encode(
            x=alt.X("Importance Score:Q", title="Normalized Linear Decision Boundary Weight"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            color=alt.Color(
                "Importance Score:Q",
                scale=alt.Scale(scheme="tealblues"),
                legend=None,
            ),
            tooltip=["Feature", "Importance Score", "Impact Level"],
        )
        .properties(height=220)
    )
    spec5 = chart5.to_dict()
    assert "data" in spec5
    assert spec5["encoding"]["color"]["scale"]["scheme"] == "tealblues"


def test_drift_reference_dataset_loading_compatibility():
    """Verify that dataset loading for the drift module works without AttributeError on ExperimentConfig."""
    from src.config import ExperimentConfig, get_default_config
    from src.data.loader import load_and_validate_dataset, split_dataset

    config = get_default_config()
    assert isinstance(config, ExperimentConfig)
    # Validate that load_and_validate_dataset and split_dataset accept the config
    X, y, feature_names, target_names = load_and_validate_dataset(config)
    assert X.shape == (150, 4)
    assert len(y) == 150

    splits = split_dataset(X, y, feature_names, target_names, config)
    assert splits.X_train.shape == (120, 4)
    assert splits.X_test.shape == (30, 4)


def test_drift_insufficient_samples_detection():
    """Verify drift detector handles empty or minimal data gracefully without crashing."""
    import pandas as pd

    from src.monitoring.drift import DataDriftDetector

    ref_df = pd.DataFrame({
        "sepal length (cm)": [5.1, 4.9, 4.7, 4.6, 5.0],
        "sepal width (cm)": [3.5, 3.0, 3.2, 3.1, 3.6],
        "petal length (cm)": [1.4, 1.4, 1.3, 1.5, 1.4],
        "petal width (cm)": [0.2, 0.2, 0.2, 0.2, 0.2],
    })
    detector = DataDriftDetector(baseline_df=ref_df)

    # Empty comparison
    empty_df = pd.DataFrame(columns=ref_df.columns)
    summary = detector.evaluate_drift(empty_df)
    assert summary is not None
    for rep in summary.feature_reports:
        assert rep.drift_status == "INSUFFICIENT_DATA"



def test_sidebar_navigation_rendering(api_client):
    """Verify sidebar navigation renders cleanly for both online and offline states."""
    from frontend.components.navigation import render_sidebar_navigation

    with patch("streamlit.sidebar"):
        with patch("streamlit.page_link") as mock_page_link, patch("httpx.Client.get") as mock_get:
            # Online state
            mock_get.return_value = MagicMock(status_code=200, json=lambda: {"model_version": "1.0.0"})
            health = render_sidebar_navigation(api_client)
            assert mock_page_link.called
            assert health.get("model_version") == "1.0.0"

            # Offline state
            mock_get.side_effect = httpx.ConnectError("Refused")
            health_offline = render_sidebar_navigation(api_client)
            assert health_offline == {}





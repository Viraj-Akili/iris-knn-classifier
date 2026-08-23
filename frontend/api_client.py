"""
FastAPI HTTP Client for Streamlit Dashboard.
Encapsulates all communication with the ML inference and observability backend.
"""

import os
from typing import Any

import httpx


class ApiClientError(Exception):
    """Base exception for API client errors."""
    pass


class ApiConnectionError(ApiClientError):
    """Raised when FastAPI backend is unreachable."""
    pass


class ApiValidationError(ApiClientError):
    """Raised on HTTP 422 validation errors."""
    pass


class ApiUnavailableError(ApiClientError):
    """Raised on HTTP 503 model unloaded errors."""
    pass


class IrisApiClient:
    """Reusable, robust HTTP client connecting frontend to FastAPI ML service."""

    def __init__(self, base_url: str | None = None, timeout: float = 5.0) -> None:
        self.base_url = (
            base_url or os.environ.get("IRIS_API_URL", "http://127.0.0.1:8008")
        ).rstrip("/")
        self.timeout = timeout

    def check_connection(self) -> bool:
        """Quick boolean ping to test backend reachability."""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/health")
                return res.status_code == 200
        except Exception:
            return False

    def get_health(self) -> dict[str, Any]:
        """Fetch liveness health status."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/health")
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError as e:
            raise ApiConnectionError(f"Cannot connect to FastAPI at {self.base_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ApiClientError(f"Health check failed with status {e.response.status_code}") from e
        except Exception as e:
            raise ApiClientError(f"Unexpected error querying /health: {e}") from e

    def get_readiness(self) -> dict[str, Any]:
        """Fetch readiness probe status."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/readiness")
                if res.status_code == 503:
                    return res.json()
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError as e:
            raise ApiConnectionError(f"Cannot connect to FastAPI at {self.base_url}: {e}") from e
        except Exception as e:
            raise ApiClientError(f"Unexpected error querying /readiness: {e}") from e

    def get_model_info(self) -> dict[str, Any]:
        """Retrieve model metadata, hyperparameters, and CV scores."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/model")
                if res.status_code == 503:
                    raise ApiUnavailableError("Model artifact is not loaded in backend memory.")
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError as e:
            raise ApiConnectionError(f"Cannot connect to FastAPI at {self.base_url}: {e}") from e
        except Exception as e:
            raise ApiClientError(f"Error querying /model: {e}") from e

    def predict(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> dict[str, Any]:
        """Send feature measurements to FastAPI for real-time model prediction."""
        payload = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/predict", json=payload)
                if res.status_code == 422:
                    error_detail = res.json().get("detail", "Validation failed")
                    raise ApiValidationError(f"Invalid input: {error_detail}")
                if res.status_code == 503:
                    raise ApiUnavailableError("Model artifact is unavailable.")
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError as e:
            raise ApiConnectionError(f"Cannot connect to FastAPI at {self.base_url}: {e}") from e
        except (ApiValidationError, ApiUnavailableError):
            raise
        except Exception as e:
            raise ApiClientError(f"Prediction failed: {e}") from e

    def get_observability_summary(self) -> dict[str, Any]:
        """Retrieve real-time observability summary from /observability/summary."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/observability/summary")
                res.raise_for_status()
                return res.json()
        except httpx.ConnectError as e:
            raise ApiConnectionError(f"Cannot connect to FastAPI at {self.base_url}: {e}") from e
        except Exception as e:
            raise ApiClientError(f"Error querying /observability/summary: {e}") from e

    def get_prometheus_metrics(self) -> str:
        """Fetch raw Prometheus text metrics from /metrics."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(
                    f"{self.base_url}/metrics",
                    headers={"Accept": "text/plain; version=0.0.4; charset=utf-8"},
                )
                res.raise_for_status()
                return res.text
        except httpx.ConnectError as e:
            raise ApiConnectionError(f"Cannot connect to FastAPI at {self.base_url}: {e}") from e
        except Exception as e:
            raise ApiClientError(f"Error querying /metrics: {e}") from e

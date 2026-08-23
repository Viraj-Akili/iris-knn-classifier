"""
Pydantic schemas for request validation, structured responses, and observability reporting.
"""

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Input payload for flower species classification."""
    sepal_length: float = Field(
        ...,
        ge=0.1,
        le=15.0,
        description="Sepal length in centimeters (valid biological range: 0.1 - 15.0 cm)",
        examples=[5.8],
    )
    sepal_width: float = Field(
        ...,
        ge=0.1,
        le=15.0,
        description="Sepal width in centimeters (valid biological range: 0.1 - 15.0 cm)",
        examples=[2.7],
    )
    petal_length: float = Field(
        ...,
        ge=0.1,
        le=15.0,
        description="Petal length in centimeters (valid biological range: 0.1 - 15.0 cm)",
        examples=[4.1],
    )
    petal_width: float = Field(
        ...,
        ge=0.1,
        le=15.0,
        description="Petal width in centimeters (valid biological range: 0.1 - 15.0 cm)",
        examples=[1.0],
    )

    @field_validator("sepal_length", "sepal_width", "petal_length", "petal_width")
    @classmethod
    def validate_finite_numbers(cls, value: float, info) -> float:
        """Reject NaN, Infinity, and non-finite numbers."""
        if math.isnan(value):
            raise ValueError(f"Feature '{info.field_name}' must be a finite number, got NaN.")
        if math.isinf(value):
            raise ValueError(f"Feature '{info.field_name}' must be a finite number, got Infinity.")
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.8,
                "sepal_width": 2.7,
                "petal_length": 4.1,
                "petal_width": 1.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Real-time prediction response payload with calibrated probabilities and latency."""
    request_id: str = Field(..., description="Unique UUID for tracing and logging")
    prediction: str = Field(..., description="Predicted flower class label")
    confidence: float = Field(..., description="Maximum predicted class probability score [0.0, 1.0]")
    probabilities: dict[str, float] = Field(
        ..., description="Calibrated probabilities per class summing to 1.0"
    )
    model_name: str = Field(..., description="Champion model identifier")
    model_version: str = Field(..., description="Semantic version of the deployed model artifact")
    inference_latency_ms: float = Field(
        ..., description="Measured real-time inference latency in milliseconds (monotonic clock)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
                "prediction": "versicolor",
                "confidence": 0.9412,
                "probabilities": {
                    "setosa": 0.0125,
                    "versicolor": 0.9412,
                    "virginica": 0.0463,
                },
                "model_name": "Support Vector Machine",
                "model_version": "1.0.0",
                "inference_latency_ms": 1.24,
            }
        }
    }


class HealthResponse(BaseModel):
    """Liveness health check response schema."""
    status: str = Field(..., description="Service liveness state: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Flag indicating if champion model is loaded in memory")
    model_version: str | None = Field(None, description="Active model version if loaded")
    service: str = Field(..., description="Service name identifier")


class ReadinessResponse(BaseModel):
    """Readiness probe response schema."""
    status: str = Field(..., description="Readiness state: 'ready' or 'not_ready'")
    model_loaded: bool = Field(..., description="True if model artifact is ready for prediction traffic")
    model_name: str | None = Field(None, description="Loaded model architecture name")
    model_version: str | None = Field(None, description="Loaded model semantic version")
    service: str = Field(..., description="Service name identifier")
    checks: dict[str, str] = Field(..., description="Subsystem readiness check details")


class ModelInfoResponse(BaseModel):
    """Metadata response schema for loaded champion model."""
    model_name: str
    model_version: str
    model_type: str
    training_timestamp: str
    feature_names: list[str]
    class_names: list[str]
    target_names: list[str] | None = None
    cv_accuracy: float
    cv_accuracy_std: float
    cv_f1_macro: float
    test_accuracy: float
    test_precision_macro: float
    test_recall_macro: float
    test_f1_macro: float
    test_f1_weighted: float
    best_hyperparameters: dict[str, Any]

    model_config = {
        "populate_by_name": True,
    }


class FeatureAggregateStat(BaseModel):
    """Privacy-safe running statistics for a single numerical input feature."""
    count: int = Field(..., description="Total count of observed samples")
    mean: float = Field(..., description="Running mean feature measurement (cm)")
    std: float = Field(..., description="Running standard deviation (cm)")
    min: float = Field(..., description="Minimum observed measurement (cm)")
    max: float = Field(..., description="Maximum observed measurement (cm)")


class ObservabilitySummaryResponse(BaseModel):
    """Comprehensive real-time dashboard observability snapshot schema."""
    service: str = Field(..., description="Service identifier")
    model_name: str = Field(..., description="Loaded model name")
    model_version: str = Field(..., description="Loaded model semantic version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    total_requests: int = Field(..., description="Total requests processed")
    successful_predictions: int = Field(..., description="Successful prediction count")
    failed_requests: int = Field(..., description="Total error/failure count")
    predictions_by_class: dict[str, int] = Field(
        ..., description="Distribution of predictions across species classes"
    )
    confidence_distribution: dict[str, int] = Field(
        ..., description="Count of predictions grouped by confidence tier (high, medium, low)"
    )
    latency_statistics_ms: dict[str, float] = Field(
        ..., description="Empirical latency percentiles and summary (mean, p50, p95, p99, min, max)"
    )
    feature_aggregates: dict[str, FeatureAggregateStat] = Field(
        ..., description="Privacy-safe running statistics for incoming features"
    )
    health_state: dict[str, str] = Field(
        ..., description="Current system liveness and readiness states"
    )
    confidence_limitation_note: str = Field(
        ...,
        description="Notice: Model confidence score does not strictly equal correctness probability.",
    )


class RuntimeMetricsResponse(BaseModel):
    """In-process runtime inference metrics schema."""
    total_requests: int = Field(..., description="Total API requests received")
    successful_predictions: int = Field(..., description="Total successful prediction inferences")
    failed_requests: int = Field(..., description="Total failed or errored requests")
    avg_latency_ms: float = Field(..., description="Average inference latency in milliseconds")
    min_latency_ms: float = Field(..., description="Minimum recorded inference latency in ms")
    max_latency_ms: float = Field(..., description="Maximum recorded inference latency in ms")
    low_confidence_predictions: int = Field(
        ..., description="Count of predictions with confidence score below threshold"
    )
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    note: str = Field(
        ...,
        description="In-process application-level runtime metrics; distinct from persistent monitoring.",
    )


class ErrorResponse(BaseModel):
    """Structured error response schema."""
    detail: str = Field(..., description="Human-readable error explanation")
    request_id: str | None = Field(None, description="Request ID associated with the error")
    error_code: str = Field(..., description="Standardized error category code")

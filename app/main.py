"""
FastAPI application for real-time machine learning inference and observability service.
"""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import AppConfig
from app.logging_config import log_inference_event, setup_logging
from app.metrics import MetricsTracker
from app.model_service import ModelInferenceError, ModelService
from app.observability import ObservabilityTracker
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    ObservabilitySummaryResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)

# Global runtime state instances
app_config = AppConfig.from_env()
logger = setup_logging()
model_service = ModelService()
metrics_tracker = MetricsTracker(low_confidence_threshold=app_config.low_confidence_threshold)
observability_tracker = ObservabilityTracker(
    service_name=app_config.service_name,
    high_confidence_threshold=app_config.high_confidence_threshold,
    low_confidence_threshold=app_config.low_confidence_threshold,
    latency_buffer_size=app_config.latency_buffer_size,
    prediction_log_path=app_config.prediction_log_path,
    enable_prediction_logging=app_config.enable_prediction_logging,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle manager: loads model once at startup and registers observability."""
    logger.info("Initializing ML inference application and observability subsystem...")
    try:
        model_service.load(
            model_path=app_config.model_artifact_path,
            metadata_path=app_config.metadata_artifact_path,
        )
        # Register model metadata in Prometheus Gauge
        classifier_type = model_service.metadata.get("model_type", "SVC")
        observability_tracker.set_model_info(
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            model_type=classifier_type,
        )
        logger.info(
            f"Successfully loaded champion model: {model_service.model_name} (v{model_service.model_version})"
        )
    except Exception as e:
        logger.critical(f"FATAL: Model loading failed during startup: {e}", exc_info=True)

    yield

    logger.info("Shutting down ML inference application...")


app = FastAPI(
    title="Iris Machine Learning Inference & Observability API",
    description=(
        "Production-style real-time ML inference service for Iris species classification. "
        "Powered by a Scikit-Learn champion model pipeline with calibrated probabilities, "
        "Prometheus metrics, latency percentile tracking, and privacy-safe feature monitoring."
    ),
    version=app_config.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    """Assign or extract request_id and inject tracing headers."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    metrics_tracker.record_request()
    observability_tracker.record_request(
        endpoint=request.url.path,
        model_version=model_service.model_version if model_service.is_loaded else "unknown",
    )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# -----------------------------------------------------------------------------
# Exception Handlers
# -----------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle invalid request schema or input data."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    metrics_tracker.record_failure()
    observability_tracker.record_failure(endpoint=request.url.path, error_code="VALIDATION_ERROR")

    errors = exc.errors()
    error_messages = []
    for err in errors:
        loc = " -> ".join(str(loc_part) for loc_part in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        error_messages.append(f"{loc}: {msg}")

    detail_str = "; ".join(error_messages)
    log_inference_event(
        logger=logger,
        request_id=request_id,
        endpoint=request.url.path,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        latency_ms=0.0,
        model_version=model_service.model_version,
        error=detail_str,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": detail_str,
            "request_id": request_id,
            "error_code": "VALIDATION_ERROR",
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle standard HTTP exceptions with structured response."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    if exc.status_code >= 400:
        metrics_tracker.record_failure()
        observability_tracker.record_failure(
            endpoint=request.url.path, error_code=f"HTTP_{exc.status_code}"
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": request_id,
            "error_code": f"HTTP_{exc.status_code}",
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions without leaking stack trace details."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    metrics_tracker.record_failure()
    observability_tracker.record_failure(
        endpoint=request.url.path, error_code="INTERNAL_SERVER_ERROR"
    )

    logger.error(
        f"Unhandled server error on {request.url.path} (request_id={request_id}): {exc}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error encountered while processing request.",
            "request_id": request_id,
            "error_code": "INTERNAL_SERVER_ERROR",
        },
        headers={"X-Request-ID": request_id},
    )


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Health Probe",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Liveness probe: verifies the application process is running."""
    is_ready = model_service.is_loaded
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_version=model_service.model_version if is_ready else None,
        service=app_config.service_name,
    )


@app.get(
    "/readiness",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    tags=["System"],
    responses={
        503: {"model": ReadinessResponse, "description": "Service not ready to receive traffic"},
    },
)
async def readiness_check() -> Any:
    """
    Readiness probe: checks if model artifact is loaded into memory and ready for traffic.
    Returns HTTP 200 when ready, HTTP 503 when model is unavailable.
    """
    is_ready = model_service.is_loaded
    checks = {
        "model_artifact": "loaded" if is_ready else "not_loaded",
        "metadata": "loaded" if is_ready else "not_loaded",
        "inference_engine": "ready" if is_ready else "offline",
    }

    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ReadinessResponse(
                status="not_ready",
                model_loaded=False,
                model_name=None,
                model_version=None,
                service=app_config.service_name,
                checks=checks,
            ).model_dump(),
        )

    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        model_name=model_service.model_name,
        model_version=model_service.model_version,
        service=app_config.service_name,
        checks=checks,
    )


@app.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Model Metadata & Performance",
    tags=["Model"],
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def get_model_info() -> ModelInfoResponse:
    """Retrieve full model metadata, training configuration, CV scores, and evaluation metrics."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not loaded or unavailable.",
        )
    return ModelInfoResponse(**model_service.get_metadata())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Real-Time Flower Species Prediction",
    tags=["Inference"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model unavailable"},
    },
)
async def predict_species(request: Request, payload: PredictionRequest) -> PredictionResponse:
    """
    Perform real-time flower species inference on input feature measurements.
    Returns predicted class label, confidence, calibrated class probabilities, and measured latency.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if not model_service.is_loaded:
        metrics_tracker.record_failure()
        observability_tracker.record_failure(endpoint="/predict", error_code="HTTP_503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded in memory.",
        )

    features_dict = payload.model_dump()

    try:
        pred_label, confidence, probas, latency_ms = model_service.predict(features_dict)

        # Record in runtime metrics tracker
        metrics_tracker.record_prediction(latency_ms=latency_ms, confidence=confidence)

        # Record in central observability tracker (Prometheus, feature stats, JSONL log)
        observability_tracker.record_prediction(
            request_id=request_id,
            features=features_dict,
            prediction=pred_label,
            confidence=confidence,
            latency_ms=latency_ms,
            model_name=model_service.model_name,
            model_version=model_service.model_version,
        )

        log_inference_event(
            logger=logger,
            request_id=request_id,
            endpoint="/predict",
            status_code=status.HTTP_200_OK,
            latency_ms=latency_ms,
            model_version=model_service.model_version,
            prediction=pred_label,
            confidence=confidence,
        )

        return PredictionResponse(
            request_id=request_id,
            prediction=pred_label,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in probas.items()},
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            inference_latency_ms=round(latency_ms, 3),
        )

    except ModelInferenceError as e:
        metrics_tracker.record_failure()
        observability_tracker.record_failure(endpoint="/predict", error_code="INFERENCE_FAILED")
        log_inference_event(
            logger=logger,
            request_id=request_id,
            endpoint="/predict",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            latency_ms=0.0,
            model_version=model_service.model_version,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference execution failed.",
        ) from e


@app.get(
    "/metrics",
    summary="Prometheus Metrics Exposition",
    tags=["Observability"],
)
async def get_prometheus_metrics(request: Request) -> Response:
    """
    Expose Prometheus-compatible metrics.
    If client requests application/json via Accept header, returns backwards-compatible JSON metrics.
    Otherwise, returns standard Prometheus text exposition format.
    """
    accept_header = request.headers.get("Accept", "")
    if "application/json" in accept_header and "text/plain" not in accept_header:
        return JSONResponse(content=metrics_tracker.get_metrics())

    metrics_bytes = observability_tracker.generate_prometheus_metrics()
    return Response(
        content=metrics_bytes,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get(
    "/observability/summary",
    response_model=ObservabilitySummaryResponse,
    summary="Real-Time Observability Dashboard Summary",
    tags=["Observability"],
)
async def get_observability_summary() -> ObservabilitySummaryResponse:
    """
    Retrieve comprehensive real-time ML observability summary for dashboards and operations.
    Includes request counts, class distributions, confidence tiers, latency percentiles,
    privacy-safe feature running statistics, and health state.
    """
    summary_dict = observability_tracker.get_summary(
        model_name=model_service.model_name if model_service.is_loaded else "Unknown",
        model_version=model_service.model_version if model_service.is_loaded else "unknown",
        is_ready=model_service.is_loaded,
        is_alive=True,
    )
    return ObservabilitySummaryResponse(**summary_dict)

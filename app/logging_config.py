"""
Structured logging configuration for production ML inference service.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured fields if provided in record
        if hasattr(record, "structured_data") and isinstance(record.structured_data, dict):
            log_entry.update(record.structured_data)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root and application loggers with JSON formatting."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicated logs
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)

    app_logger = logging.getLogger("iris_inference_api")
    app_logger.setLevel(level)
    return app_logger


def log_inference_event(
    logger: logging.Logger,
    request_id: str,
    endpoint: str,
    status_code: int,
    latency_ms: float,
    model_version: str,
    prediction: str | None = None,
    confidence: float | None = None,
    error: str | None = None,
) -> None:
    """Structured inference log helper."""
    structured_payload: dict[str, Any] = {
        "request_id": request_id,
        "endpoint": endpoint,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 3),
        "model_version": model_version,
    }

    if prediction is not None:
        structured_payload["prediction"] = prediction
    if confidence is not None:
        structured_payload["confidence"] = round(confidence, 4)
    if error is not None:
        structured_payload["error"] = error

    extra = {"structured_data": structured_payload}

    if status_code >= 500:
        logger.error(f"Inference error on {endpoint}", extra=extra)
    elif status_code >= 400:
        logger.warning(f"Client error on {endpoint}", extra=extra)
    else:
        logger.info(f"Inference completed on {endpoint}", extra=extra)

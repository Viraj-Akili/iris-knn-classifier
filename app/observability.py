"""
Real-time ML Observability, Prometheus metrics, latency percentile tracking,
and privacy-safe feature aggregation module.
"""

import collections
import json
import math
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class OnlineFeatureStats:
    """
    Welford's algorithm for numerically stable, privacy-safe online calculation
    of mean, variance, standard deviation, min, and max for a single continuous feature.
    Does NOT store raw sample points.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from the mean
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, x: float) -> None:
        """Add a single observation x."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if x < self.min_val:
            self.min_val = x
        if x > self.max_val:
            self.max_val = x

    @property
    def variance(self) -> float:
        return self.M2 / self.count if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": round(self.mean, 4) if self.count > 0 else 0.0,
            "std": round(self.std, 4) if self.count > 0 else 0.0,
            "min": round(self.min_val, 4) if self.min_val != float("inf") else 0.0,
            "max": round(self.max_val, 4) if self.max_val != float("-inf") else 0.0,
        }


class ObservabilityTracker:
    """
    Central observability engine coordinating:
    1. Prometheus metrics exposition
    2. Bounded latency aggregation for p50/p95/p99 percentiles
    3. Prediction and confidence tier distributions
    4. Privacy-safe aggregate feature monitoring
    5. Structured JSONL prediction logging
    """

    def __init__(
        self,
        service_name: str = "iris-ml-api",
        high_confidence_threshold: float = 0.90,
        low_confidence_threshold: float = 0.70,
        latency_buffer_size: int = 1000,
        prediction_log_path: Path | None = None,
        enable_prediction_logging: bool = True,
    ) -> None:
        self.service_name = service_name
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.latency_buffer_size = latency_buffer_size
        self.prediction_log_path = prediction_log_path
        self.enable_prediction_logging = enable_prediction_logging

        self._lock = threading.Lock()
        self._start_time = time.time()
        self._total_requests = 0
        self._successful_predictions = 0
        self._failed_requests = 0

        # Distributions
        self._predictions_by_class: dict[str, int] = collections.defaultdict(int)
        self._confidence_distribution: dict[str, int] = {
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        # Bounded sliding window for exact empirical latency percentiles
        self._latency_buffer: collections.deque = collections.deque(maxlen=latency_buffer_size)

        # Privacy-safe feature aggregators
        self._feature_stats: dict[str, OnlineFeatureStats] = {
            "sepal_length": OnlineFeatureStats(),
            "sepal_width": OnlineFeatureStats(),
            "petal_length": OnlineFeatureStats(),
            "petal_width": OnlineFeatureStats(),
        }

        # Initialize dedicated Prometheus Registry
        self._init_prometheus_metrics()

    def _init_prometheus_metrics(self) -> None:
        """Create Prometheus metric collectors in a dedicated registry."""
        self.registry = CollectorRegistry()

        self.prom_requests_total = Counter(
            "iris_prediction_requests_total",
            "Total HTTP prediction requests received",
            ["endpoint", "model_version"],
            registry=self.registry,
        )
        self.prom_success_total = Counter(
            "iris_prediction_success_total",
            "Total successful prediction inferences",
            ["model_version"],
            registry=self.registry,
        )
        self.prom_failures_total = Counter(
            "iris_prediction_failures_total",
            "Total failed API requests",
            ["endpoint", "error_code"],
            registry=self.registry,
        )
        self.prom_predictions_by_class = Counter(
            "iris_predictions_by_class_total",
            "Total predictions count segmented by predicted species class",
            ["predicted_class", "model_version"],
            registry=self.registry,
        )
        self.prom_predictions_by_confidence = Counter(
            "iris_predictions_by_confidence_tier_total",
            "Total predictions count segmented by confidence tier (high, medium, low)",
            ["tier", "model_version"],
            registry=self.registry,
        )
        self.prom_latency_seconds = Histogram(
            "iris_prediction_latency_seconds",
            "Inference latency in seconds",
            ["model_version"],
            buckets=(0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry,
        )
        self.prom_model_info = Gauge(
            "iris_model_info",
            "Active champion model metadata",
            ["model_name", "model_version", "model_type"],
            registry=self.registry,
        )

    def set_model_info(self, model_name: str, model_version: str, model_type: str) -> None:
        """Register active model metadata with Prometheus Gauge."""
        with self._lock:
            self.prom_model_info.labels(
                model_name=model_name,
                model_version=model_version,
                model_type=model_type,
            ).set(1.0)

    def record_request(self, endpoint: str = "/predict", model_version: str = "1.0.0") -> None:
        """Increment request counters."""
        with self._lock:
            self._total_requests += 1
            self.prom_requests_total.labels(endpoint=endpoint, model_version=model_version).inc()

    def record_prediction(
        self,
        request_id: str,
        features: dict[str, float],
        prediction: str,
        confidence: float,
        latency_ms: float,
        model_name: str,
        model_version: str,
    ) -> None:
        """
        Record a successful prediction inference across metrics, distributions,
        latency buffer, feature statistics, and JSONL log.
        """
        # Determine confidence tier
        if confidence >= self.high_confidence_threshold:
            tier = "high"
        elif confidence >= self.low_confidence_threshold:
            tier = "medium"
        else:
            tier = "low"

        latency_sec = latency_ms / 1000.0

        with self._lock:
            self._successful_predictions += 1
            self._predictions_by_class[prediction] += 1
            self._confidence_distribution[tier] += 1
            self._latency_buffer.append(latency_ms)

            # Update Prometheus metrics
            self.prom_success_total.labels(model_version=model_version).inc()
            self.prom_predictions_by_class.labels(
                predicted_class=prediction, model_version=model_version
            ).inc()
            self.prom_predictions_by_confidence.labels(
                tier=tier, model_version=model_version
            ).inc()
            self.prom_latency_seconds.labels(model_version=model_version).observe(latency_sec)

            # Update privacy-safe online feature running stats
            for feat_name, feat_val in features.items():
                if feat_name in self._feature_stats:
                    self._feature_stats[feat_name].update(feat_val)

        # Asynchronously or safely append to predictions.jsonl
        if self.enable_prediction_logging and self.prediction_log_path:
            self._append_prediction_log(
                request_id=request_id,
                model_name=model_name,
                model_version=model_version,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
            )

    def record_failure(self, endpoint: str = "/predict", error_code: str = "ERROR") -> None:
        """Record a failed or rejected inference request."""
        with self._lock:
            self._failed_requests += 1
            self.prom_failures_total.labels(endpoint=endpoint, error_code=error_code).inc()

    def _append_prediction_log(
        self,
        request_id: str,
        model_name: str,
        model_version: str,
        prediction: str,
        confidence: float,
        latency_ms: float,
    ) -> None:
        """Append a privacy-conscious single-line JSONL prediction event to disk."""
        try:
            log_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "model_name": model_name,
                "model_version": model_version,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "latency_ms": round(latency_ms, 3),
            }
            log_path = Path(self.prediction_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            # Observability logging should never crash the main request loop
            pass

    def get_latency_statistics(self) -> dict[str, float]:
        """Calculate exact empirical percentiles from the bounded sliding latency buffer."""
        with self._lock:
            if not self._latency_buffer:
                return {
                    "mean_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                }

            arr = np.array(self._latency_buffer)
            return {
                "mean_ms": round(float(np.mean(arr)), 3),
                "p50_ms": round(float(np.median(arr)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
                "min_ms": round(float(np.min(arr)), 3),
                "max_ms": round(float(np.max(arr)), 3),
            }

    def get_summary(
        self,
        model_name: str = "Unknown",
        model_version: str = "1.0.0",
        is_ready: bool = True,
        is_alive: bool = True,
    ) -> dict[str, Any]:
        """Generate a complete runtime observability summary for dashboards and APIs."""
        with self._lock:
            uptime = time.time() - self._start_time
            pred_classes = dict(self._predictions_by_class)
            conf_dist = dict(self._confidence_distribution)
            feat_aggs = {k: v.to_dict() for k, v in self._feature_stats.items()}
            tot_req = self._total_requests
            succ_req = self._successful_predictions
            fail_req = self._failed_requests

        latency_stats = self.get_latency_statistics()

        return {
            "service": self.service_name,
            "model_name": model_name,
            "model_version": model_version,
            "uptime_seconds": round(uptime, 2),
            "total_requests": tot_req,
            "successful_predictions": succ_req,
            "failed_requests": fail_req,
            "predictions_by_class": pred_classes,
            "confidence_distribution": conf_dist,
            "latency_statistics_ms": latency_stats,
            "feature_aggregates": feat_aggs,
            "health_state": {
                "liveness": "healthy" if is_alive else "unhealthy",
                "readiness": "ready" if is_ready else "not_ready",
            },
            "confidence_limitation_note": (
                "Note: Model confidence score reflects calibrated softmax/Platt output "
                "and does not strictly equal ground-truth correctness probability."
            ),
        }

    def generate_prometheus_metrics(self) -> bytes:
        """Expose collected metrics in standard Prometheus text format."""
        return generate_latest(self.registry)

    def reset(self) -> None:
        """Reset internal metrics and buffers (used for test isolation)."""
        with self._lock:
            self._start_time = time.time()
            self._total_requests = 0
            self._successful_predictions = 0
            self._failed_requests = 0
            self._predictions_by_class.clear()
            self._confidence_distribution = {"high": 0, "medium": 0, "low": 0}
            self._latency_buffer.clear()
            self._feature_stats = {
                "sepal_length": OnlineFeatureStats(),
                "sepal_width": OnlineFeatureStats(),
                "petal_length": OnlineFeatureStats(),
                "petal_width": OnlineFeatureStats(),
            }
            self._init_prometheus_metrics()

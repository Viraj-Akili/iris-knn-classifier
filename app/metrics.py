"""
Thread-safe in-process runtime metrics tracker for inference requests.
"""

import threading
import time
from typing import Any


class MetricsTracker:
    """
    In-process runtime inference metrics collector.
    Thread-safe counter and latency tracking for real-time diagnostics.
    """

    def __init__(self, low_confidence_threshold: float = 0.70):
        self.low_confidence_threshold = low_confidence_threshold
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._total_requests = 0
        self._successful_predictions = 0
        self._failed_requests = 0
        self._low_confidence_predictions = 0
        self._total_latency_ms = 0.0
        self._min_latency_ms = float("inf")
        self._max_latency_ms = 0.0

    def record_request(self) -> None:
        """Increment overall request count."""
        with self._lock:
            self._total_requests += 1

    def record_prediction(self, latency_ms: float, confidence: float) -> None:
        """Record a successful prediction and its measured latency and confidence."""
        with self._lock:
            self._successful_predictions += 1
            self._total_latency_ms += latency_ms
            if latency_ms < self._min_latency_ms:
                self._min_latency_ms = latency_ms
            if latency_ms > self._max_latency_ms:
                self._max_latency_ms = latency_ms
            if confidence < self.low_confidence_threshold:
                self._low_confidence_predictions += 1

    def record_failure(self) -> None:
        """Increment failed request counter."""
        with self._lock:
            self._failed_requests += 1

    def get_metrics(self) -> dict[str, Any]:
        """Retrieve snapshot of runtime metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            avg_latency = (
                self._total_latency_ms / self._successful_predictions
                if self._successful_predictions > 0
                else 0.0
            )
            min_lat = (
                self._min_latency_ms
                if self._min_latency_ms != float("inf")
                else 0.0
            )

            return {
                "total_requests": self._total_requests,
                "successful_predictions": self._successful_predictions,
                "failed_requests": self._failed_requests,
                "avg_latency_ms": round(avg_latency, 3),
                "min_latency_ms": round(min_lat, 3),
                "max_latency_ms": round(self._max_latency_ms, 3),
                "low_confidence_predictions": self._low_confidence_predictions,
                "uptime_seconds": round(uptime, 2),
                "note": (
                    "In-process application runtime metrics. "
                    "Distinct from persistent production monitoring (Prometheus / Grafana)."
                ),
            }

    def reset(self) -> None:
        """Reset counters (used in testing)."""
        with self._lock:
            self._start_time = time.time()
            self._total_requests = 0
            self._successful_predictions = 0
            self._failed_requests = 0
            self._low_confidence_predictions = 0
            self._total_latency_ms = 0.0
            self._min_latency_ms = float("inf")
            self._max_latency_ms = 0.0

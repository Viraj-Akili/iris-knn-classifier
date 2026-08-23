"""
Comprehensive Multi-Level Inference Latency & Observability Overhead Benchmark.
Measures:
  A. Raw Model-Only In-Memory Inference
  B. FastAPI HTTP Request Baseline (without observability logging)
  C. FastAPI HTTP Request with Full Observability (Prometheus, Percentile Tracking, Online Feature Stats, JSONL Logging)
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient

from app.config import AppConfig
from app.main import app, observability_tracker
from app.model_service import ModelService

SAMPLE_PAYLOAD = {
    "sepal_length": 5.8,
    "sepal_width": 2.7,
    "petal_length": 4.1,
    "petal_width": 1.0,
}


def benchmark_model_only(n_iterations: int = 500) -> tuple[np.ndarray, str, str]:
    """Level A: Benchmark raw ModelService in-memory prediction latency."""
    config = AppConfig.from_env()
    service = ModelService()
    service.load(config.model_artifact_path, config.metadata_artifact_path)

    # Warm-up
    for _ in range(25):
        service.predict(SAMPLE_PAYLOAD)

    latencies_ms: list[float] = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        service.predict(SAMPLE_PAYLOAD)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    return np.array(latencies_ms), service.model_name, service.model_version


def benchmark_api_without_observability(n_iterations: int = 250) -> np.ndarray:
    """Level B: Benchmark FastAPI HTTP latency with prediction logging disabled."""
    prev_logging = observability_tracker.enable_prediction_logging
    observability_tracker.enable_prediction_logging = False

    latencies_ms: list[float] = []
    with TestClient(app) as client:
        # Warmup
        for _ in range(15):
            client.post("/predict", json=SAMPLE_PAYLOAD)

        for _ in range(n_iterations):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=SAMPLE_PAYLOAD)
            t1 = time.perf_counter()
            assert resp.status_code == 200
            latencies_ms.append((t1 - t0) * 1000.0)

    observability_tracker.enable_prediction_logging = prev_logging
    return np.array(latencies_ms)


def benchmark_api_with_full_observability(n_iterations: int = 250) -> np.ndarray:
    """Level C: Benchmark FastAPI HTTP latency with full Prometheus, percentile tracking, and JSONL logging."""
    observability_tracker.enable_prediction_logging = True

    latencies_ms: list[float] = []
    with TestClient(app) as client:
        # Warmup
        for _ in range(15):
            client.post("/predict", json=SAMPLE_PAYLOAD)

        for _ in range(n_iterations):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=SAMPLE_PAYLOAD)
            t1 = time.perf_counter()
            assert resp.status_code == 200
            latencies_ms.append((t1 - t0) * 1000.0)

    return np.array(latencies_ms)


def print_stats_table(title: str, arr: np.ndarray, n: int) -> None:
    """Helper to format latency summary."""
    print("=" * 70)
    print(f"  {title} (N = {n})")
    print("=" * 70)
    print(f"  Mean Latency     : {np.mean(arr):.3f} ms")
    print(f"  Median (p50)     : {np.median(arr):.3f} ms")
    print(f"  p90 Latency      : {np.percentile(arr, 90):.3f} ms")
    print(f"  p95 Latency      : {np.percentile(arr, 95):.3f} ms")
    print(f"  p99 Latency      : {np.percentile(arr, 99):.3f} ms")
    print(f"  Min Latency      : {np.min(arr):.3f} ms")
    print(f"  Max Latency      : {np.max(arr):.3f} ms")
    print(f"  Throughput (QPS) : {1000.0 / np.mean(arr):.1f} req/sec")
    print("=" * 70)


def main() -> None:
    # Mute noisy test logs during benchmark run
    logging.getLogger("iris_inference_api").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print("\n" + "#" * 70)
    print("  MULTI-TIER ML INFERENCE & OBSERVABILITY PERFORMANCE PROFILER")
    print("#" * 70 + "\n")

    # Level A
    arr_a, model_name, model_version = benchmark_model_only(500)
    print_stats_table(f"TIER A: RAW MODEL-ONLY INFERENCE [{model_name} v{model_version}]", arr_a, 500)
    print()

    # Level B
    arr_b = benchmark_api_without_observability(250)
    print_stats_table("TIER B: FASTAPI HTTP /predict (BASE INFERENCE)", arr_b, 250)
    print()

    # Level C
    arr_c = benchmark_api_with_full_observability(250)
    print_stats_table("TIER C: FASTAPI HTTP /predict (FULL OBSERVABILITY + LOGGING)", arr_c, 250)
    print()

    # Comparative Delta
    overhead_ms = np.mean(arr_c) - np.mean(arr_b)
    print("=" * 70)
    print("  OBSERVABILITY OVERHEAD SUMMARY")
    print("=" * 70)
    print(f"  Raw Model Inference Mean  : {np.mean(arr_a):.3f} ms")
    print(f"  FastAPI Base HTTP Mean    : {np.mean(arr_b):.3f} ms")
    print(f"  FastAPI Observability Mean: {np.mean(arr_c):.3f} ms")
    print(f"  Net Observability Overhead: +{overhead_ms:.3f} ms ({(overhead_ms / np.mean(arr_b)) * 100:.1f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

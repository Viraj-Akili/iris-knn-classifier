#!/usr/bin/env python3
"""
Production Cloud Smoke Test & Live Verification Suite.
Executes end-to-end integration probes, multi-class predictions,
observability checks, and latency profiling against deployed services.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ProductionSmokeTester:
    """Verifies health, predictions, observability, and latency of production deployment."""

    def __init__(self, api_url: str, frontend_url: str | None = None, timeout: float = 15.0):
        self.api_url = api_url.rstrip("/")
        self.frontend_url = frontend_url.rstrip("/") if frontend_url else None
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)
        self.results: dict[str, bool] = {}

    def log(self, section: str, message: str, status: str = "INFO") -> None:
        symbols = {"INFO": "*", "SUCCESS": "+", "WARN": "!", "FAIL": "X"}
        sym = symbols.get(status, "*")
        print(f"[{sym}] [{status:7s}] [{section:18s}] {message}")

    def test_frontend_health(self) -> bool:
        """Probe frontend web server for health and availability."""
        if not self.frontend_url:
            self.log("FRONTEND", "No frontend URL provided. Skipping frontend probe.", "WARN")
            return True

        self.log("FRONTEND", f"Probing frontend at {self.frontend_url}...")
        try:
            # First try Streamlit health probe, fallback to root
            try:
                res = self.client.get(f"{self.frontend_url}/_stcore/health")
                if res.status_code == 200:
                    self.log("FRONTEND", f"Streamlit health probe OK (HTTP {res.status_code})", "SUCCESS")
                    return True
            except Exception:
                pass

            res_root = self.client.get(f"{self.frontend_url}/")
            if res_root.status_code == 200:
                self.log("FRONTEND", f"Frontend root page OK (HTTP {res_root.status_code})", "SUCCESS")
                return True
            else:
                self.log("FRONTEND", f"Frontend returned HTTP {res_root.status_code}", "FAIL")
                return False
        except Exception as e:
            self.log("FRONTEND", f"Connection to frontend failed: {e}", "FAIL")
            return False

    def test_backend_liveness(self) -> bool:
        """Probe GET /health endpoint."""
        url = f"{self.api_url}/health"
        self.log("BACKEND /health", f"Requesting GET {url}...")
        try:
            start_t = time.perf_counter()
            res = self.client.get(url)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            if res.status_code == 200:
                data = res.json()
                self.log("BACKEND /health", f"Status: {data.get('status')}, Model Loaded: {data.get('model_loaded')} ({elapsed_ms:.1f} ms)", "SUCCESS")
                return True
            else:
                self.log("BACKEND /health", f"Unexpected status code {res.status_code}: {res.text}", "FAIL")
                return False
        except Exception as e:
            self.log("BACKEND /health", f"Failed to connect: {e}", "FAIL")
            return False

    def test_backend_readiness(self) -> bool:
        """Probe GET /readiness endpoint."""
        url = f"{self.api_url}/readiness"
        self.log("BACKEND /readiness", f"Requesting GET {url}...")
        try:
            start_t = time.perf_counter()
            res = self.client.get(url)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            if res.status_code == 200:
                data = res.json()
                self.log("BACKEND /readiness", f"Status: {data.get('status')}, Model: {data.get('model_name')} v{data.get('model_version')} ({elapsed_ms:.1f} ms)", "SUCCESS")
                return True
            else:
                self.log("BACKEND /readiness", f"Readiness check failed with HTTP {res.status_code}: {res.text}", "FAIL")
                return False
        except Exception as e:
            self.log("BACKEND /readiness", f"Failed to connect: {e}", "FAIL")
            return False

    def test_backend_model_info(self) -> bool:
        """Probe GET /model endpoint."""
        url = f"{self.api_url}/model"
        self.log("BACKEND /model", f"Requesting GET {url}...")
        try:
            res = self.client.get(url)
            if res.status_code == 200:
                data = res.json()
                self.log("BACKEND /model", f"Model: {data.get('model_name')} | CV Accuracy: {data.get('cv_accuracy', 0)*100:.2f}% | Holdout: {data.get('test_accuracy', 0)*100:.2f}%", "SUCCESS")
                return True
            else:
                self.log("BACKEND /model", f"HTTP {res.status_code}: {res.text}", "FAIL")
                return False
        except Exception as e:
            self.log("BACKEND /model", f"Connection error: {e}", "FAIL")
            return False

    def test_predictions(self) -> bool:
        """Execute real predictions across all target species and boundary case."""
        url = f"{self.api_url}/predict"
        test_cases = [
            {
                "name": "Typical Setosa",
                "payload": {"sepal_length": 5.0, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                "expected": "setosa",
            },
            {
                "name": "Typical Versicolor",
                "payload": {"sepal_length": 5.9, "sepal_width": 2.8, "petal_length": 4.2, "petal_width": 1.3},
                "expected": "versicolor",
            },
            {
                "name": "Typical Virginica",
                "payload": {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 2.0},
                "expected": "virginica",
            },
            {
                "name": "Boundary Specimen (#134)",
                "payload": {"sepal_length": 6.1, "sepal_width": 2.6, "petal_length": 5.6, "petal_width": 1.4},
                "expected": "virginica",
            },
        ]

        all_passed = True
        self.log("INFERENCE", "Executing multi-class verification against POST /predict...")

        for case in test_cases:
            try:
                start_t = time.perf_counter()
                res = self.client.post(url, json=case["payload"])
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0

                if res.status_code == 200:
                    data = res.json()
                    pred = data.get("prediction")
                    conf = data.get("confidence", 0.0)
                    trace_id = data.get("request_id")
                    server_lat = data.get("inference_latency_ms", 0.0)

                    match_sym = "[MATCH]" if pred == case["expected"] else "[DIFF]"
                    self.log(
                        "INFERENCE",
                        f"{case['name']:25s} -> Pred: {pred:10s} (Conf: {conf*100:5.1f}%) | HTTP: {elapsed_ms:5.1f}ms (Server: {server_lat:.2f}ms) | Trace: {trace_id[:8]}... {match_sym}",
                        "SUCCESS" if pred == case["expected"] else "WARN",
                    )
                else:
                    self.log("INFERENCE", f"{case['name']} failed with HTTP {res.status_code}: {res.text}", "FAIL")
                    all_passed = False
            except Exception as e:
                self.log("INFERENCE", f"{case['name']} error: {e}", "FAIL")
                all_passed = False

        return all_passed

    def test_observability(self) -> bool:
        """Verify GET /observability/summary and GET /metrics."""
        all_passed = True

        # Observability summary
        summary_url = f"{self.api_url}/observability/summary"
        try:
            res = self.client.get(summary_url)
            if res.status_code == 200:
                data = res.json()
                total = data.get("total_requests", 0)
                lat = data.get("latency_percentiles", {})
                p50 = lat.get("p50_ms", 0.0)
                p95 = lat.get("p95_ms", 0.0)
                self.log("OBSERVABILITY", f"Telemetry summary OK | Total Requests: {total} | p50: {p50:.2f}ms | p95: {p95:.2f}ms", "SUCCESS")
            else:
                self.log("OBSERVABILITY", f"Summary failed with HTTP {res.status_code}", "FAIL")
                all_passed = False
        except Exception as e:
            self.log("OBSERVABILITY", f"Summary request error: {e}", "FAIL")
            all_passed = False

        # Prometheus metrics
        metrics_url = f"{self.api_url}/metrics"
        try:
            res = self.client.get(metrics_url)
            if res.status_code == 200 and "iris_prediction_requests_total" in res.text:
                self.log("OBSERVABILITY", f"Prometheus exposition OK ({len(res.text)} bytes)", "SUCCESS")
            else:
                self.log("OBSERVABILITY", f"Prometheus exposition invalid (HTTP {res.status_code})", "FAIL")
                all_passed = False
        except Exception as e:
            self.log("OBSERVABILITY", f"Prometheus request error: {e}", "FAIL")
            all_passed = False

        return all_passed

    def profile_latency(self, num_samples: int = 25) -> dict[str, Any]:
        """Profile live network and inference latency."""
        url = f"{self.api_url}/predict"
        payload = {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 4.1, "petal_width": 1.0}

        self.log("LATENCY PROFILE", f"Profiling latency across {num_samples} HTTP inferences...")
        latencies_ms: list[float] = []

        # Warm-up / cold-start measurement
        try:
            t0 = time.perf_counter()
            self.client.post(url, json=payload)
            cold_start_ms = (time.perf_counter() - t0) * 1000.0
            self.log("LATENCY PROFILE", f"Initial Request Latency (Cold Start / Warmup): {cold_start_ms:.2f} ms", "INFO")
        except Exception as e:
            self.log("LATENCY PROFILE", f"Warmup request failed: {e}", "FAIL")
            return {}

        for _ in range(num_samples):
            try:
                t0 = time.perf_counter()
                res = self.client.post(url, json=payload)
                if res.status_code == 200:
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            except Exception:
                pass

        if not latencies_ms:
            self.log("LATENCY PROFILE", "Failed to collect latency samples.", "FAIL")
            return {}

        import numpy as np
        lat_arr = np.array(latencies_ms)
        stats = {
            "samples": len(lat_arr),
            "cold_start_ms": cold_start_ms,
            "mean_ms": float(np.mean(lat_arr)),
            "median_ms": float(np.median(lat_arr)),
            "p90_ms": float(np.percentile(lat_arr, 90)),
            "p95_ms": float(np.percentile(lat_arr, 95)),
            "p99_ms": float(np.percentile(lat_arr, 99)),
            "min_ms": float(np.min(lat_arr)),
            "max_ms": float(np.max(lat_arr)),
        }

        self.log(
            "LATENCY PROFILE",
            f"Mean: {stats['mean_ms']:.2f}ms | p50: {stats['median_ms']:.2f}ms | p95: {stats['p95_ms']:.2f}ms | p99: {stats['p99_ms']:.2f}ms (N={stats['samples']})",
            "SUCCESS",
        )
        return stats

    def run_all(self, num_latency_samples: int = 25) -> bool:
        """Run complete smoke test suite."""
        print("\n" + "=" * 75)
        print("  PRODUCTION CLOUD SMOKE TEST & VERIFICATION SUITE")
        print(f"  Target API URL     : {self.api_url}")
        print(f"  Target Frontend URL: {self.frontend_url or 'N/A'}")
        print(f"  Timeout            : {self.timeout}s")
        print("=" * 75 + "\n")

        self.results["frontend"] = self.test_frontend_health()
        self.results["liveness"] = self.test_backend_liveness()
        self.results["readiness"] = self.test_backend_readiness()
        self.results["model_info"] = self.test_backend_model_info()
        self.results["predictions"] = self.test_predictions()
        self.results["observability"] = self.test_observability()

        # Run latency profile
        self.profile_latency(num_samples=num_latency_samples)

        print("\n" + "=" * 75)
        print("  SMOKE TEST EXECUTION SUMMARY")
        print("=" * 75)
        overall_success = True
        for check, passed in self.results.items():
            status_str = "PASSED [OK]" if passed else "FAILED [X]"
            print(f"  * {check:25s}: {status_str}")
            if not passed:
                overall_success = False

        print("=" * 75)
        if overall_success:
            print("  *** ALL PRODUCTION PROBES & VERIFICATIONS PASSED SUCCESSFULLY! ***")
        else:
            print("  [X] ONE OR MORE VERIFICATION PROBES FAILED!")
        print("=" * 75 + "\n")

        return overall_success


def main():
    parser = argparse.ArgumentParser(description="Production Cloud Deployment Smoke Test Suite")
    parser.add_argument(
        "--api-url",
        default=os.getenv("IRIS_API_URL", "http://127.0.0.1:8008"),
        help="Base URL of the FastAPI inference backend (e.g., https://iris-ml-backend.onrender.com)",
    )
    parser.add_argument(
        "--frontend-url",
        default=os.getenv("IRIS_FRONTEND_URL", "http://127.0.0.1:8501"),
        help="Base URL of the Streamlit frontend (e.g., https://iris-ml-frontend.onrender.com)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of latency benchmark iterations",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP request timeout in seconds (default: 15.0s)",
    )

    args = parser.parse_args()
    tester = ProductionSmokeTester(api_url=args.api_url, frontend_url=args.frontend_url, timeout=args.timeout)
    success = tester.run_all(num_latency_samples=args.samples)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

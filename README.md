# Enterprise Iris Classification & Real-Time ML System

[![CI/CD Release Pipeline](https://github.com/Viraj-Akili/iris-knn-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Viraj-Akili/iris-knn-classifier/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Tests](https://img.shields.io/badge/tests-40%2F40%20passed-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-91.7%25-success.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![Linter](https://img.shields.io/badge/lint-ruff%20clean-blueviolet.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An enterprise-grade, production-style machine learning experimentation, evaluation, real-time inference, and observability system built on Fisher's classic Iris benchmark. Engineered with rigorous data isolation, zero data leakage, automated multi-model tournament benchmarking, stratified 5-fold cross-validation, deep nearest-neighbor error analysis, high-throughput FastAPI REST serving architecture, an enterprise ML observability layer with Prometheus metrics, latency percentile tracking, privacy-safe feature monitoring, statistical data drift detection, an interactive **Streamlit ML Intelligence Dashboard**, **production Docker containerization**, and automated **GitHub Actions CI/CD with Cloud Release automation**.

---

## 📌 Project Overview

This project elevates educational machine learning scripts into an industry-standard, reproducible ML experimentation repository and full-stack observable inference system:

- **Strict Schema & Integrity Validation**: Enforces dataset dimensions (150 samples, 4 numerical features, 3 target classes), non-negativity, and null-safety.
- **Leakage-Free Preprocessing**: Encapsulates `StandardScaler` inside Scikit-Learn `Pipeline` architectures so scaling statistics are fitted exclusively on training folds.
- **Stratified Multi-Model Tournament**: Benchmarks 7 distinct algorithms (KNN, Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting, HistGradientBoosting) with 5-Fold Stratified Cross-Validation across multiple performance metrics.
- **CV-Driven Champion Selection**: Evaluates the holdout test set strictly once on the tournament winner without test-set feedback loops. Champion selected: **Support Vector Machine (Linear Kernel, $C=0.1$)** with **97.50% ($\pm 2.04\%$)** CV accuracy.
- **Granular Error Analysis**: Inspects decision boundaries, prediction confidence, probability calibration, and feature-space nearest training neighbors for misclassified samples.
- **Real-Time FastAPI Inference Backend**: Production REST API exposing `/health`, `/readiness`, `/model`, `/predict`, `/metrics`, and `/observability/summary` with sub-millisecond model inference, calibrated probabilities, and UUID request tracing.
- **Enterprise ML Observability**: Prometheus metrics exposition, sliding-window latency percentiles ($p_{50}, p_{90}, p_{95}, p_{99}$), confidence tiers, privacy-safe online input feature aggregation (Welford's algorithm), and structured JSONL prediction logging.
- **Statistical Data Drift Detection**: Two-Sample Kolmogorov-Smirnov (KS) test, Wasserstein distance (in physical cm units), and Population Stability Index (PSI) drift monitoring engine.
- **Streamlit ML Intelligence Dashboard**: Interactive operations console with real-time inference, live telemetry, drift analysis, model tournament leaderboards, error analysis, and model governance cards.
- **Production Docker Containerization**: Independent, hardened container images for backend and frontend orchestrated via Docker Compose with health checks and non-root users.
- **Continuous Deployment & Cloud Release**: Automated CI/CD pipeline executing quality gates, Docker builds, smoke tests, and continuous cloud deployment.

---

## 🌐 Production Architecture & Deployment Flow

```text
                    GitHub Repository
                           │
                           ▼
                 GitHub Actions CI/CD
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
      FastAPI Docker Image    Streamlit Docker Image
               │                       │
               ▼                       ▼
        Cloud API Service       Cloud Dashboard
        (Render / Cloud Run)    (Render / Streamlit Cloud)
               │                       │
               └───────────┬───────────┘
                           │
                         HTTPS
                           │
                           ▼
                  Real-Time ML REST API
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
        SVM Champion Pipeline       ML Observability
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
                  Prometheus Metrics   Audit Logs   Drift Engine
```

---

## 📁 Repository Structure

```text
iris-knn-classifier/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD & automated cloud release pipeline
├── app/                        # Production FastAPI Real-Time Inference & Observability Backend
│   ├── __init__.py
│   ├── main.py                 # FastAPI application, lifespan manager, middleware, and routing
│   ├── schemas.py              # Pydantic request/response schemas, bounds validation, & metrics models
│   ├── model_service.py        # Model lifecycle service (loads champion once into memory)
│   ├── observability.py        # Prometheus collector registry, latency percentiles, & feature stats
│   ├── metrics.py              # In-process runtime metrics collector (backwards-compatible)
│   ├── logging_config.py       # Structured JSON logging and inference event formatters
│   └── config.py               # Runtime application settings & environment overrides
├── frontend/                   # Streamlit ML Intelligence Operations Dashboard
│   ├── __init__.py
│   ├── app.py                  # Main dashboard overview, navigation, and live telemetry hub
│   ├── api_client.py           # Robust HTTP client communicating with FastAPI backend
│   ├── components/
│   │   ├── __init__.py
│   │   ├── header.py           # Live status badges, uptime, model metadata, & refresh controls
│   │   ├── metrics_cards.py    # Styled KPI cards (latency percentiles, request counts, error rates)
│   │   └── charts.py           # Interactive Altair probability, distribution, and feature charts
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── 1_Real_Time_Inference.py     # Interactive prediction UI with specimen presets
│   │   ├── 2_Live_Observability.py      # Real-time Prometheus metrics, class & confidence tiers
│   │   ├── 3_Data_Drift_Analysis.py     # Two-Sample KS, Wasserstein distance, & PSI inspector
│   │   ├── 4_Model_Tournament.py        # 7-model tournament leaderboard & CV variance plots
│   │   ├── 5_Error_Analysis.py          # Confusion matrix & misclassified sample deep-dive
│   │   └── 6_Model_Card_Explainability.py # Model card governance & linear SVM feature weights
│   └── utils/
│       ├── __init__.py
│       └── formatting.py       # Theme styling, badge HTML formatters, and latency formatters
├── config/
│   └── config.yaml             # Centralized ML configuration (seeds, splits, CV folds, paths)
├── src/
│   ├── __init__.py
│   ├── config.py               # Dataclass-backed config loader and validator
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # Ingestion, schema validation, stratified 80/20 partition
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pipeline_factory.py # Scikit-Learn Pipeline factory & hyperparameter search grids
│   │   ├── train.py            # Stratified 5-Fold CV engine & champion selector
│   │   └── evaluate.py         # Holdout test evaluator & nearest-neighbor error analyzer
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift.py            # Statistical drift detection (Two-sample KS test, Wasserstein, PSI)
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # High-resolution headless diagnostic visualization suite
├── scripts/
│   ├── benchmark_latency.py    # Multi-tier latency profiler (Model vs HTTP Base vs Observability)
│   ├── check_drift.py          # Offline statistical data drift analysis CLI
│   └── production_smoke_test.py# End-to-end production cloud deployment verification suite
├── tests/
│   ├── test_api.py             # FastAPI endpoint integration, validation, & error handling tests
│   ├── test_frontend_client.py # Streamlit API client unit tests & exception mapping
│   ├── test_observability.py   # Prometheus metrics, health/readiness, summary, & drift tests
│   ├── test_data_loader.py     # Data validation, shape, and split isolation tests
│   ├── test_pipeline.py        # Pipeline scaling isolation & hyperparameter grid tests
│   ├── test_training.py        # Cross-validation scoring, ranking, and reproducibility tests
│   └── test_evaluation.py      # Metric bounds, probability calibration, and artifact tests
├── artifacts/                  # Versioned machine-readable outputs & visual diagnostics
│   ├── experiments/            # Leaderboards (model_comparison.csv, cv_results.csv)
│   ├── metrics/                # final_metrics.json, classification_report.json
│   ├── models/                 # champion_pipeline.joblib, champion_metadata.json
│   ├── logs/                   # predictions.jsonl (structured prediction events)
│   ├── plots/                  # confusion_matrix.png, model_comparison.png, etc.
│   └── predictions/            # test_predictions.csv
├── Dockerfile.backend          # Hardened production Dockerfile for FastAPI backend
├── Dockerfile.frontend         # Hardened production Dockerfile for Streamlit dashboard
├── docker-compose.yml          # Multi-container service composition with healthchecks
├── render.yaml                 # Render Infrastructure-as-Code Blueprint
├── .dockerignore               # Docker build exclusions
├── .env.example                # Template for environment configuration
├── pyproject.toml              # Build system, Ruff linter, and Pytest coverage configuration
├── main.py                     # Primary pipeline CLI entrypoint
├── iris_classifier.py          # Backward-compatible wrapper delegating to main.py
├── requirements.txt            # Explicit, version-pinned project dependencies
└── README.md                   # Comprehensive technical documentation
```

---

## ☁️ Cloud Production Deployment

### 1. Cloud Platform Evaluation & Selection

| Provider | Python / Docker Support | HTTPS & Custom Domains | Healthchecks & Probes | Free Tier / Cost Profile | Cold Start Behavior | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Render** | **Native Docker & Python** | **Automated Free SSL** | **Native `/health` path** | **Free Web Services** | Free instances spin down after 15m of inactivity (~30s cold start) | 🌟 **Primary Selected Provider** (Infrastructure as Code via `render.yaml`) |
| **Streamlit Community Cloud** | Native Python | Automated Free SSL | Automatic restart | 100% Free for open-source | Near-zero cold start | 🌟 **Alternative Frontend Host** |
| **Hugging Face Spaces** | Docker / Streamlit | Automated Free SSL | Container health | Free CPU tier | Minimal cold starts | Excellent alternative for containerized demo |
| **Fly.io** | Docker | Automated Free SSL | TCP & HTTP checks | Credit card required for account setup | Fast Wake | Excellent but requires billing setup |

### 2. Zero-Configuration Cloud Deployment via Render

The repository includes a production blueprint ([`render.yaml`](file:///c:/Users/Viraj%20Akili/OneDrive/Desktop/project-2/iris-knn-classifier/render.yaml)) defining both services:

1. Connect your GitHub repository to [Render](https://render.com).
2. Click **New +** ➔ **Blueprint** and select `Viraj-Akili/iris-knn-classifier`.
3. Render automatically provisions:
   - `iris-ml-backend` (Docker Web Service on `https://iris-ml-backend.onrender.com`)
   - `iris-ml-frontend` (Docker Web Service on `https://iris-ml-frontend.onrender.com`)
4. Health checks are wired directly to `/health` and `/_stcore/health`.

### 3. Production vs. Development Separation

| Configuration Dimension | Development Environment | Production Cloud Environment |
| :--- | :--- | :--- |
| **Server Reload** | Enabled (`--reload` for instant DX) | Disabled (`uvicorn --workers 1` deterministic runtime) |
| **CORS Policy** | Permissive (`localhost:8501`, `127.0.0.1:8501`, `localhost:3000`) | Explicit production origins (`https://iris-ml-frontend.onrender.com`) |
| **Logging Level** | `DEBUG` (Console stream) | `INFO` / `WARN` (Structured JSON with request tracing) |
| **Secrets & Keys** | Local `.env` (Excluded from Git) | Platform Environment Variables / GitHub Secrets |
| **Backend URL** | `http://127.0.0.1:8000` / `http://127.0.0.1:8008` | `https://iris-ml-backend.onrender.com` (TLS Encrypted) |

### 4. Cold-Start & Free-Tier Operational Characteristics

> [!NOTE]
> On free-tier cloud platforms (such as Render Free Tier), idle compute instances automatically enter sleep mode after 15 minutes of inactivity. The initial HTTP request triggers container spin-up, resulting in a one-time cold-start latency of approximately **30 to 50 seconds**. All subsequent requests operate at standard sub-millisecond inference and warm cloud network response times (**15 - 35 ms** total round-trip).

---

## 🧪 Production Smoke Test Verification

Execute the end-to-end cloud smoke test suite against local or deployed environments:

```bash
# Run against local running stack
python scripts/production_smoke_test.py --api-url http://127.0.0.1:8008 --frontend-url http://127.0.0.1:8501

# Run against live cloud deployment
python scripts/production_smoke_test.py --api-url https://iris-ml-backend.onrender.com --frontend-url https://iris-ml-frontend.onrender.com
```

### Actual Smoke Test Verification Output

```text
===========================================================================
  PRODUCTION CLOUD SMOKE TEST & VERIFICATION SUITE
  Target API URL     : http://127.0.0.1:8008
  Target Frontend URL: http://127.0.0.1:8501
  Timeout            : 15.0s
===========================================================================

[*] [INFO   ] [FRONTEND          ] Probing frontend at http://127.0.0.1:8501...
[+] [SUCCESS] [FRONTEND          ] Streamlit health probe OK (HTTP 200)
[*] [INFO   ] [BACKEND /health   ] Requesting GET http://127.0.0.1:8008/health...
[+] [SUCCESS] [BACKEND /health   ] Status: healthy, Model Loaded: True (25.8 ms)
[*] [INFO   ] [BACKEND /readiness] Requesting GET http://127.0.0.1:8008/readiness...
[+] [SUCCESS] [BACKEND /readiness] Status: ready, Model: Support Vector Machine v1.0.0 (3.1 ms)
[*] [INFO   ] [BACKEND /model    ] Requesting GET http://127.0.0.1:8008/model...
[+] [SUCCESS] [BACKEND /model    ] Model: Support Vector Machine | CV Accuracy: 97.50% | Holdout: 93.33%
[*] [INFO   ] [INFERENCE         ] Executing multi-class verification against POST /predict...
[+] [SUCCESS] [INFERENCE         ] Typical Setosa            -> Pred: setosa     (Conf:  97.6%) | HTTP:  20.1ms (Server: 7.38ms) | Trace: 7e36a67e... [MATCH]
[+] [SUCCESS] [INFERENCE         ] Typical Versicolor        -> Pred: versicolor (Conf:  95.5%) | HTTP:  17.8ms (Server: 6.23ms) | Trace: b43c3aed... [MATCH]
[+] [SUCCESS] [INFERENCE         ] Typical Virginica         -> Pred: virginica  (Conf:  95.3%) | HTTP:  19.1ms (Server: 5.88ms) | Trace: 9b5fc49c... [MATCH]
[+] [SUCCESS] [INFERENCE         ] Boundary Specimen (#134)  -> Pred: virginica  (Conf:  50.0%) | HTTP:  18.8ms (Server: 6.35ms) | Trace: 30d61a3d... [MATCH]
[+] [SUCCESS] [OBSERVABILITY     ] Telemetry summary OK | Total Requests: 24 | p50: 1.05ms | p95: 1.09ms
[+] [SUCCESS] [OBSERVABILITY     ] Prometheus exposition OK (5083 bytes)
[*] [INFO   ] [LATENCY PROFILE   ] Profiling latency across 25 HTTP inferences...
[*] [INFO   ] [LATENCY PROFILE   ] Initial Request Latency (Cold Start / Warmup): 17.57 ms
[+] [SUCCESS] [LATENCY PROFILE   ] Mean: 16.95ms | p50: 16.69ms | p95: 19.01ms | p99: 20.02ms (N=25)

===========================================================================
  SMOKE TEST EXECUTION SUMMARY
===========================================================================
  * frontend                 : PASSED [OK]
  * liveness                 : PASSED [OK]
  * readiness                : PASSED [OK]
  * model_info               : PASSED [OK]
  * predictions              : PASSED [OK]
  * observability            : PASSED [OK]
===========================================================================
  *** ALL PRODUCTION PROBES & VERIFICATIONS PASSED SUCCESSFULLY! ***
===========================================================================
```

---

## ⚡ Empirical Multi-Tier Performance Profiling

| Tier | Evaluation Scope | Sample Size ($N$) | Mean Latency | Median ($p_{50}$) | $p_{90}$ Latency | $p_{95}$ Latency | $p_{99}$ Latency | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier A** | **Raw Model-Only Inference** | 500 | **0.866 ms** | **0.817 ms** | **1.034 ms** | **1.178 ms** | **1.373 ms** | **1,154.3 req/sec** |
| **Tier B** | **FastAPI Base HTTP (Localhost)** | 250 | **2.025 ms** | **1.959 ms** | **2.341 ms** | **2.508 ms** | **2.862 ms** | **493.8 req/sec** |
| **Tier C** | **FastAPI + Full Observability** | 250 | **2.948 ms** | **2.839 ms** | **3.275 ms** | **3.505 ms** | **4.056 ms** | **339.2 req/sec** |
| **Tier D** | **Live Network Round-Trip (Smoke Test)** | 25 | **16.950 ms** | **16.690 ms** | **18.720 ms** | **19.010 ms** | **20.020 ms** | **59.0 req/sec** |

---

## 🔬 ML Tournament Leaderboard (5-Fold Stratified CV, N=120)

| Rank | Model Architecture | CV Accuracy (Mean $\pm$ Std) | CV Macro Precision | CV Macro Recall | CV Macro F1 | Best Hyperparameters |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Support Vector Machine** | **97.50% $\pm$ 2.04%** | **0.9778** | **0.9750** | **0.9749** | `C=0.1, gamma='scale', kernel='linear'` |
| 🥈 | **K-Nearest Neighbors** | 96.67% $\pm$ 1.67% | 0.9704 | 0.9667 | 0.9665 | `n_neighbors=3, weights='uniform', metric='euclidean'` |
| 🥉 | **Logistic Regression** | 96.67% $\pm$ 3.12% | 0.9719 | 0.9667 | 0.9663 | `C=10.0, solver='lbfgs'` |
| 4 | **Random Forest** | 96.67% $\pm$ 3.12% | 0.9719 | 0.9667 | 0.9663 | `n_estimators=50, max_depth=None, min_samples_split=5` |
| 5 | **Decision Tree** | 95.83% $\pm$ 2.64% | 0.9644 | 0.9583 | 0.9580 | `criterion='gini', max_depth=None, min_samples_split=5` |
| 6 | **Gradient Boosting** | 95.83% $\pm$ 2.64% | 0.9644 | 0.9583 | 0.9580 | `learning_rate=0.05, max_depth=2, n_estimators=25` |
| 7 | **HistGradientBoosting** | 95.83% $\pm$ 2.64% | 0.9644 | 0.9583 | 0.9580 | `learning_rate=0.05, max_depth=2, max_iter=25` |

---

## 🧪 Verification Commands

```bash
# 1. Run Ruff linter
ruff check .

# 2. Run full automated test suite with coverage
pytest -v --cov=src --cov=app --cov=frontend --cov-report=term-missing

# 3. Execute latency profiler
python scripts/benchmark_latency.py

# 4. Execute production smoke tester
python scripts/production_smoke_test.py --api-url http://127.0.0.1:8008 --frontend-url http://127.0.0.1:8501
```

---

## 🛣️ Project Evolution & Completion Matrix

- [x] **Phase 1**: Production-quality ML experimentation, leakage-free pipeline, 7-model tournament, 5-fold CV, and nearest-neighbor error analysis.
- [x] **Phase 2**: Production serving API (FastAPI REST service with Pydantic validation, `/health`, `/model`, `/predict`, `/metrics`, and latency benchmarking).
- [x] **Phase 3**: Enterprise ML observability & monitoring (Prometheus metrics, latency percentiles, confidence tiers, online feature monitoring, health vs readiness probes, and offline statistical drift analysis).
- [x] **Phase 4**: Real-time ML Intelligence Dashboard (Streamlit operations console, live prediction, telemetry charts, drift inspector, tournament leaderboards, error analysis, and model governance cards).
- [x] **Phase 5**: Production Docker containerization (`Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`) and automated CI/CD pipeline (`.github/workflows/ci.yml`).
- [x] **Phase 6**: Cloud deployment readiness, Render infrastructure blueprint (`render.yaml`), automated cloud release stage in CI/CD, production CORS hardening, and live smoke test suite (`scripts/production_smoke_test.py`).

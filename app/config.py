"""
Configuration management for the FastAPI real-time ML inference service and observability layer.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Application runtime settings, artifact paths, and observability thresholds."""
    service_name: str = "iris-ml-api"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Path configurations
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    model_artifact_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "models" / "champion_pipeline.joblib"
    )
    metadata_artifact_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "models" / "champion_metadata.json"
    )
    prediction_log_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "logs" / "predictions.jsonl"
    )

    # Security & CORS
    allowed_origins: list[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ])

    # Observability & Confidence Tiers
    high_confidence_threshold: float = 0.90
    low_confidence_threshold: float = 0.70
    latency_buffer_size: int = 1000
    enable_prediction_logging: bool = True

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build AppConfig with overrides from environment variables."""
        config = cls()

        if "API_HOST" in os.environ:
            config.host = os.environ["API_HOST"]
        if "API_PORT" in os.environ:
            config.port = int(os.environ["API_PORT"])
        if "API_DEBUG" in os.environ:
            config.debug = os.environ["API_DEBUG"].lower() in ("true", "1", "yes")
        if "MODEL_ARTIFACT_PATH" in os.environ:
            config.model_artifact_path = Path(os.environ["MODEL_ARTIFACT_PATH"])
        if "METADATA_ARTIFACT_PATH" in os.environ:
            config.metadata_artifact_path = Path(os.environ["METADATA_ARTIFACT_PATH"])
        if "PREDICTION_LOG_PATH" in os.environ:
            config.prediction_log_path = Path(os.environ["PREDICTION_LOG_PATH"])
        if "CORS_ORIGINS" in os.environ:
            origins = [o.strip() for o in os.environ["CORS_ORIGINS"].split(",") if o.strip()]
            if origins:
                config.allowed_origins = origins
        if "HIGH_CONFIDENCE_THRESHOLD" in os.environ:
            config.high_confidence_threshold = float(os.environ["HIGH_CONFIDENCE_THRESHOLD"])
        if "LOW_CONFIDENCE_THRESHOLD" in os.environ:
            config.low_confidence_threshold = float(os.environ["LOW_CONFIDENCE_THRESHOLD"])
        if "ENABLE_PREDICTION_LOGGING" in os.environ:
            config.enable_prediction_logging = os.environ["ENABLE_PREDICTION_LOGGING"].lower() in ("true", "1", "yes")

        return config

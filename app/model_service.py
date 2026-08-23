"""
Model service responsible for loading the persisted champion pipeline and performing real-time inference.
"""

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class ModelLoadError(Exception):
    """Raised when the model artifact or metadata cannot be loaded."""
    pass


class ModelInferenceError(Exception):
    """Raised when an error occurs during model prediction."""
    pass


class ModelService:
    """
    Production model service encapsulating the loaded champion pipeline.
    Loads artifacts once into memory during application startup.
    """

    def __init__(self) -> None:
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.feature_names: list[str] = []
        self.target_names: list[str] = []
        self.model_version: str = "unknown"
        self.model_name: str = "unknown"
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Return True if model and metadata are loaded and ready."""
        return self._is_loaded

    def load(self, model_path: Path, metadata_path: Path) -> None:
        """
        Load serialized champion pipeline and metadata.
        Raises ModelLoadError if artifacts are missing, corrupt, or invalid.
        """
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)

        if not model_path.exists():
            raise ModelLoadError(f"Model artifact not found at: {model_path}")
        if not metadata_path.exists():
            raise ModelLoadError(f"Model metadata not found at: {metadata_path}")

        try:
            with open(metadata_path, encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise ModelLoadError(f"Failed to parse metadata JSON from {metadata_path}: {e}") from e

        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to deserialize model pipeline from {model_path}: {e}") from e

        self.feature_names = self.metadata.get(
            "feature_names",
            [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ],
        )
        self.target_names = self.metadata.get(
            "target_names",
            ["setosa", "versicolor", "virginica"],
        )
        self.model_version = self.metadata.get("model_version", "1.0.0")
        self.model_name = self.metadata.get("model_name", "Champion Model")
        self._is_loaded = True

    def predict(
        self,
        features: dict[str, float],
    ) -> tuple[str, float, dict[str, float], float]:
        """
        Perform real-time single-sample inference.

        Parameters
        ----------
        features : Dict[str, float]
            Input mapping with keys: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'.

        Returns
        -------
        Tuple[str, float, Dict[str, float], float]
            - predicted_class (str): Class name
            - confidence (float): Maximum probability [0.0, 1.0]
            - probabilities (Dict[str, float]): Calibrated probability map per class
            - latency_ms (float): Real measured inference latency in milliseconds
        """
        if not self._is_loaded or self.model is None:
            raise ModelInferenceError("ModelService is not loaded. Call load() first.")

        # Map API input keys to model's expected feature column names
        row_data = {
            "sepal length (cm)": features["sepal_length"],
            "sepal width (cm)": features["sepal_width"],
            "petal length (cm)": features["petal_length"],
            "petal width (cm)": features["petal_width"],
        }
        input_df = pd.DataFrame([row_data], columns=self.feature_names)

        # Measure real inference latency with monotonic clock
        start_time = time.perf_counter()
        try:
            # Check if model supports predict_proba
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(input_df)[0]
                pred_idx = int(np.argmax(probas))
                confidence = float(probas[pred_idx])
                probabilities = {
                    self.target_names[i]: float(probas[i])
                    for i in range(len(self.target_names))
                }
                pred_label = self.target_names[pred_idx]
            else:
                pred = self.model.predict(input_df)[0]
                pred_idx = int(pred) if isinstance(pred, (int, np.integer)) else self.target_names.index(str(pred))
                pred_label = self.target_names[pred_idx]
                confidence = 1.0
                probabilities = {name: (1.0 if name == pred_label else 0.0) for name in self.target_names}

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return pred_label, confidence, probabilities, elapsed_ms

        except Exception as e:
            raise ModelInferenceError(f"Inference execution failed: {e}") from e

    def get_metadata(self) -> dict[str, Any]:
        """Return the loaded model metadata dictionary."""
        if not self._is_loaded:
            raise ModelLoadError("Model metadata is not loaded.")
        meta = dict(self.metadata)
        meta["class_names"] = self.target_names
        meta["target_names"] = self.target_names
        return meta

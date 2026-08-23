"""
Model evaluation, test set metrics generation, and error analysis module.
"""

import json
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import NearestNeighbors

from ..config import ExperimentConfig, get_default_config
from ..data.loader import DatasetSplits
from .train import BenchmarkResult


@dataclass
class MisclassifiedSample:
    """Detailed diagnostic representation of an erroneously classified test sample."""

    sample_index: int
    test_partition_index: int
    actual_label: str
    predicted_label: str
    confidence: float
    features: dict[str, float]
    probabilities: dict[str, float]
    nearest_neighbors_context: list[dict[str, Any]] | None = None


@dataclass
class ErrorAnalysisResult:
    """Summary of test partition errors and diagnostic profiles."""

    total_test_samples: int
    correct_count: int
    misclassified_count: int
    error_rate: float
    misclassified_samples: list[MisclassifiedSample]


def perform_nearest_neighbors_lookup(
    champion_pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_names: list[str],
    query_features: pd.DataFrame,
    n_neighbors: int = 5,
) -> list[dict[str, Any]]:
    """
    Look up nearest training neighbors in the standardized space to diagnose
    boundary confusion.
    """
    try:
        # Check if scaler is in pipeline
        if "scaler" in champion_pipeline.named_steps and champion_pipeline.named_steps["scaler"] != "passthrough":
            scaler = champion_pipeline.named_steps["scaler"]
            X_train_transformed = scaler.transform(X_train)
            query_transformed = scaler.transform(query_features)
        else:
            X_train_transformed = X_train.values
            query_transformed = query_features.values

        nn_model = NearestNeighbors(n_neighbors=min(n_neighbors, len(X_train)))
        nn_model.fit(X_train_transformed)
        distances, indices = nn_model.kneighbors(query_transformed)

        neighbors_info = []
        for dist, idx in zip(distances[0], indices[0]):
            train_idx = int(X_train.index[idx])
            train_target = int(y_train.iloc[idx])
            train_label = target_names[train_target]
            train_feat = {k: float(v) for k, v in X_train.iloc[idx].to_dict().items()}
            neighbors_info.append(
                {
                    "train_sample_id": train_idx,
                    "distance": round(float(dist), 4),
                    "true_class": train_label,
                    "features": train_feat,
                }
            )
        return neighbors_info
    except Exception as e:
        return [{"error": f"Nearest neighbors calculation skipped: {str(e)}"}]


def evaluate_champion_model(
    champion_result: BenchmarkResult,
    splits: DatasetSplits,
    config: ExperimentConfig | None = None,
) -> tuple[dict[str, Any], ErrorAnalysisResult, pd.DataFrame]:
    """
    Evaluate the selected champion model on the untouched test partition,
    conduct deep error analysis, and persist evaluation artifacts.
    """
    cfg = config or get_default_config()
    cfg.paths.ensure_directories()

    model = champion_result.best_estimator
    target_names = splits.target_names
    feature_names = splits.feature_names

    # Predict on untouched test partition
    y_pred = model.predict(splits.X_test)

    # Probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(splits.X_test)
    elif hasattr(model, "decision_function"):
        df_scores = model.decision_function(splits.X_test)
        # Softmax normalization for decision function
        exp_scores = np.exp(df_scores - np.max(df_scores, axis=1, keepdims=True))
        y_proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    else:
        # Uniform dummy probabilities fallback
        n_classes = len(target_names)
        y_proba = np.ones((len(splits.X_test), n_classes)) / n_classes

    # Compute comprehensive performance metrics
    acc = float(accuracy_score(splits.y_test, y_pred))
    prec_macro = float(precision_score(splits.y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(splits.y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(splits.y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(splits.y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(splits.y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(splits.y_test, y_pred, average="weighted", zero_division=0))

    report_dict = classification_report(
        splits.y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(splits.y_test, y_pred).tolist()

    final_metrics = {
        "champion_model": champion_result.model_name,
        "best_hyperparameters": champion_result.best_params,
        "cv_accuracy_mean": champion_result.cv_metrics.accuracy_mean,
        "cv_accuracy_std": champion_result.cv_metrics.accuracy_std,
        "cv_f1_macro_mean": champion_result.cv_metrics.f1_macro_mean,
        "cv_f1_weighted_mean": champion_result.cv_metrics.f1_weighted_mean,
        "test_accuracy": acc,
        "test_precision_macro": prec_macro,
        "test_precision_weighted": prec_weighted,
        "test_recall_macro": rec_macro,
        "test_recall_weighted": rec_weighted,
        "test_f1_macro": f1_macro,
        "test_f1_weighted": f1_weighted,
        "test_sample_count": len(splits.y_test),
        "confusion_matrix": cm,
        "class_labels": target_names,
    }

    # Build predictions DataFrame
    pred_df = splits.X_test.copy()
    pred_df["actual_target"] = splits.y_test.values
    pred_df["actual_species"] = [target_names[i] for i in splits.y_test.values]
    pred_df["predicted_target"] = y_pred
    pred_df["predicted_species"] = [target_names[i] for i in y_pred]
    pred_df["is_correct"] = pred_df["actual_target"] == pred_df["predicted_target"]

    for idx, class_name in enumerate(target_names):
        pred_df[f"proba_{class_name}"] = y_proba[:, idx]

    pred_df["confidence"] = np.max(y_proba, axis=1)

    # Error Analysis
    misclassified_list: list[MisclassifiedSample] = []
    error_indices = np.where(splits.y_test.values != y_pred)[0]

    for test_idx in error_indices:
        original_idx = int(splits.X_test.index[test_idx])
        actual_cls = target_names[splits.y_test.iloc[test_idx]]
        pred_cls = target_names[y_pred[test_idx]]
        conf = float(np.max(y_proba[test_idx]))
        feat_dict = {k: float(v) for k, v in splits.X_test.iloc[test_idx].to_dict().items()}
        proba_dict = {
            target_names[i]: float(y_proba[test_idx, i])
            for i in range(len(target_names))
        }

        # Query features as a single-row DataFrame
        query_row = splits.X_test.iloc[[test_idx]]
        neighbors_ctx = perform_nearest_neighbors_lookup(
            model,
            splits.X_train,
            splits.y_train,
            target_names,
            query_row,
            n_neighbors=5,
        )

        misclassified_list.append(
            MisclassifiedSample(
                sample_index=original_idx,
                test_partition_index=int(test_idx),
                actual_label=actual_cls,
                predicted_label=pred_cls,
                confidence=conf,
                features=feat_dict,
                probabilities=proba_dict,
                nearest_neighbors_context=neighbors_ctx,
            )
        )

    error_analysis = ErrorAnalysisResult(
        total_test_samples=len(splits.y_test),
        correct_count=len(splits.y_test) - len(error_indices),
        misclassified_count=len(error_indices),
        error_rate=float(len(error_indices) / len(splits.y_test)),
        misclassified_samples=misclassified_list,
    )

    # Save artifacts
    metrics_path = cfg.paths.metrics_dir / "final_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    report_path = cfg.paths.metrics_dir / "classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    predictions_path = cfg.paths.predictions_dir / "test_predictions.csv"
    pred_df.to_csv(predictions_path, index=True)

    # Serialize model artifact
    model_save_path = cfg.paths.models_dir / "champion_pipeline.joblib"
    joblib.dump(model, model_save_path)

    from datetime import datetime

    classifier_step = model.named_steps.get("classifier", model)
    model_type = type(classifier_step).__name__

    metadata = {
        "model_name": champion_result.model_name,
        "model_version": "1.0.0",
        "model_type": model_type,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "feature_names": feature_names,
        "target_names": target_names,
        "cv_accuracy": champion_result.cv_metrics.accuracy_mean,
        "cv_accuracy_std": champion_result.cv_metrics.accuracy_std,
        "cv_f1_macro": champion_result.cv_metrics.f1_macro_mean,
        "test_accuracy": acc,
        "test_precision_macro": prec_macro,
        "test_recall_macro": rec_macro,
        "test_f1_macro": f1_macro,
        "test_f1_weighted": f1_weighted,
        "best_hyperparameters": champion_result.best_params,
    }
    with open(cfg.paths.models_dir / "champion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return final_metrics, error_analysis, pred_df

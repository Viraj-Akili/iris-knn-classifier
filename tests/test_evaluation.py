"""
Unit tests for model evaluation, prediction shapes, probability boundaries, and error analysis.
"""

from pathlib import Path

import numpy as np

from src.config import ExperimentConfig
from src.data.loader import load_and_validate_dataset, split_dataset
from src.models.evaluate import evaluate_champion_model
from src.models.train import ModelBenchmarkEngine


def test_evaluation_shapes_probabilities_and_artifacts(tmp_path: Path):
    """Verify evaluation outputs, probability axioms, and persisted artifacts."""
    from src.config import PathsConfig

    test_paths = PathsConfig(
        artifacts_dir=tmp_path,
        experiments_dir=tmp_path / "experiments",
        metrics_dir=tmp_path / "metrics",
        models_dir=tmp_path / "models",
        plots_dir=tmp_path / "plots",
        predictions_dir=tmp_path / "predictions",
    )
    config = ExperimentConfig(random_seed=42, paths=test_paths)
    X, y, feature_names, target_names = load_and_validate_dataset(config)
    splits = split_dataset(X, y, feature_names, target_names, config)

    from src.models.pipeline_factory import get_candidate_models_and_grids

    engine = ModelBenchmarkEngine(config)
    candidates = get_candidate_models_and_grids(42)
    champion_result = engine.evaluate_candidate_model(
        "K-Nearest Neighbors",
        candidates["K-Nearest Neighbors"]["pipeline"],
        candidates["K-Nearest Neighbors"]["param_grid"],
        splits.X_train,
        splits.y_train,
    )

    final_metrics, error_analysis, pred_df = evaluate_champion_model(
        champion_result=champion_result,
        splits=splits,
        config=config,
    )

    # 1. Prediction DataFrame shapes
    assert len(pred_df) == 30
    assert "actual_target" in pred_df.columns
    assert "predicted_target" in pred_df.columns
    assert "confidence" in pred_df.columns

    # 2. Probability verification (must sum to ~1.0 for each row)
    proba_cols = [f"proba_{c}" for c in target_names]
    for col in proba_cols:
        assert (pred_df[col] >= 0.0).all() and (pred_df[col] <= 1.0).all()

    proba_sums = pred_df[proba_cols].sum(axis=1)
    np.testing.assert_allclose(proba_sums, 1.0, atol=1e-5)

    # 3. Metrics validation
    assert 0.0 <= final_metrics["test_accuracy"] <= 1.0
    assert 0.0 <= final_metrics["test_f1_weighted"] <= 1.0
    assert len(final_metrics["confusion_matrix"]) == 3
    assert len(final_metrics["confusion_matrix"][0]) == 3

    # 4. Error analysis consistency
    misclassified_count = len(pred_df[pred_df["actual_target"] != pred_df["predicted_target"]])
    assert error_analysis.misclassified_count == misclassified_count
    assert error_analysis.correct_count + error_analysis.misclassified_count == 30

    # 5. Check artifact existence
    assert (config.paths.metrics_dir / "final_metrics.json").exists()
    assert (config.paths.metrics_dir / "classification_report.json").exists()
    assert (config.paths.predictions_dir / "test_predictions.csv").exists()
    assert (config.paths.models_dir / "champion_pipeline.joblib").exists()

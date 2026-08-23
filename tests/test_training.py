"""
Unit tests for multi-model benchmark training, cross-validation, and champion selection.
"""

from src.config import ExperimentConfig
from src.data.loader import load_and_validate_dataset, split_dataset
from src.models.train import ModelBenchmarkEngine


def test_benchmark_engine_cv_and_ranking():
    """Verify that the benchmark engine scores candidate models and ranks correctly."""
    from src.models.pipeline_factory import get_candidate_models_and_grids

    config = ExperimentConfig(random_seed=42)
    X, y, feature_names, target_names = load_and_validate_dataset(config)
    splits = split_dataset(X, y, feature_names, target_names, config)

    all_candidates = get_candidate_models_and_grids(42)
    # Test tournament mechanics on subset of candidate models for fast unit testing
    test_candidates = {
        "K-Nearest Neighbors": all_candidates["K-Nearest Neighbors"],
        "Logistic Regression": all_candidates["Logistic Regression"],
    }

    engine = ModelBenchmarkEngine(config)
    results, comparison_df, champion_result = engine.run_benchmark(
        splits.X_train, splits.y_train, candidate_models=test_candidates
    )

    # Verify candidate models are evaluated
    assert len(results) == 2
    assert len(comparison_df) == 2

    # Verify metrics lie in valid bounds [0.0, 1.0]
    for _, row in comparison_df.iterrows():
        assert 0.0 <= row["CV Accuracy Mean"] <= 1.0
        assert 0.0 <= row["CV F1 Macro"] <= 1.0
        assert 0.0 <= row["CV Precision Macro"] <= 1.0
        assert 0.0 <= row["CV Recall Macro"] <= 1.0

    # Verify sorting order (descending by CV accuracy)
    acc_series = comparison_df["CV Accuracy Mean"].tolist()
    assert acc_series == sorted(acc_series, reverse=True)

    # Verify champion result matches top row
    assert champion_result.model_name == comparison_df.iloc[0]["Model"]
    assert champion_result.best_estimator is not None


def test_training_reproducibility():
    """Verify that identical random seeds yield identical CV scores and hyperparameters."""
    from src.models.pipeline_factory import get_candidate_models_and_grids

    config1 = ExperimentConfig(random_seed=42)
    X1, y1, f1, t1 = load_and_validate_dataset(config1)
    splits1 = split_dataset(X1, y1, f1, t1, config1)
    engine1 = ModelBenchmarkEngine(config1)
    candidates1 = get_candidate_models_and_grids(42)

    res1 = engine1.evaluate_candidate_model(
        "K-Nearest Neighbors",
        candidates1["K-Nearest Neighbors"]["pipeline"],
        candidates1["K-Nearest Neighbors"]["param_grid"],
        splits1.X_train,
        splits1.y_train,
    )

    config2 = ExperimentConfig(random_seed=42)
    X2, y2, f2, t2 = load_and_validate_dataset(config2)
    splits2 = split_dataset(X2, y2, f2, t2, config2)
    engine2 = ModelBenchmarkEngine(config2)
    candidates2 = get_candidate_models_and_grids(42)

    res2 = engine2.evaluate_candidate_model(
        "K-Nearest Neighbors",
        candidates2["K-Nearest Neighbors"]["pipeline"],
        candidates2["K-Nearest Neighbors"]["param_grid"],
        splits2.X_train,
        splits2.y_train,
    )

    assert res1.best_params == res2.best_params
    assert res1.cv_metrics.accuracy_mean == res2.cv_metrics.accuracy_mean
    assert res1.cv_metrics.f1_macro_mean == res2.cv_metrics.f1_macro_mean

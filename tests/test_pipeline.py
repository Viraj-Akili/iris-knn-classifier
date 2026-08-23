"""
Unit tests for pipeline factory and preprocessing encapsulation.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.models.pipeline_factory import (
    get_candidate_models_and_grids,
    get_model_pipeline,
)


def test_get_model_pipeline_scaling_isolation():
    """Verify scaler is correctly fitted on train data inside pipeline without leakage."""
    estimator = KNeighborsClassifier(n_neighbors=3)
    pipeline = get_model_pipeline("KNN", estimator, use_scaler=True)

    assert "scaler" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    # Mock synthetic data
    X_train = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[50.0, 500.0]])

    pipeline.fit(X_train, y_train)

    # Scaler mean should reflect only X_train
    scaler = pipeline.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, [20.0, 200.0])

    preds = pipeline.predict(X_test)
    assert len(preds) == 1


def test_candidate_models_and_grids_completeness():
    """Verify that all required candidate models and grids are registered."""
    candidates = get_candidate_models_and_grids(random_seed=42)
    expected_models = {
        "K-Nearest Neighbors",
        "Logistic Regression",
        "Support Vector Machine",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "HistGradientBoosting",
    }

    assert set(candidates.keys()) == expected_models

    for _name, config in candidates.items():
        assert "pipeline" in config
        assert "param_grid" in config
        assert isinstance(config["param_grid"], dict)
        assert len(config["param_grid"]) > 0

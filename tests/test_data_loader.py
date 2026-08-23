"""
Unit tests for data loading, validation, and stratified splitting.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import ExperimentConfig
from src.data.loader import (
    load_and_validate_dataset,
    split_dataset,
    validate_iris_dataframe,
)


def test_load_and_validate_dataset_shapes():
    """Verify that dataset loads with exact expected shape and invariants."""
    config = ExperimentConfig()
    X, y, feature_names, target_names = load_and_validate_dataset(config)

    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert len(feature_names) == 4
    assert len(target_names) == 3
    assert set(target_names) == {"setosa", "versicolor", "virginica"}
    assert not X.isnull().any().any()
    assert (X > 0).all().all()


def test_validate_iris_dataframe_exceptions():
    """Verify data validation catches corrupted data formats."""
    # Test wrong row count
    invalid_rows = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})
    with pytest.raises(ValueError, match="Expected 150 rows"):
        validate_iris_dataframe(invalid_rows, expected_samples=150)

    # Test null values
    df_with_null = pd.DataFrame(
        np.ones((150, 5)), columns=["f1", "f2", "f3", "f4", "target"]
    )
    df_with_null["target"] = np.repeat([0, 1, 2], 50)
    df_with_null.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="contains null/NaN values"):
        validate_iris_dataframe(df_with_null, expected_samples=150, expected_features=4)

    # Test negative feature values
    df_with_neg = pd.DataFrame(
        np.ones((150, 5)), columns=["f1", "f2", "f3", "f4", "target"]
    )
    df_with_neg["target"] = np.repeat([0, 1, 2], 50)
    df_with_neg.iloc[0, 0] = -1.5
    with pytest.raises(ValueError, match="contains non-positive values"):
        validate_iris_dataframe(df_with_neg, expected_samples=150, expected_features=4)


def test_split_dataset_stratification_and_isolation():
    """Verify exact 80/20 stratified split proportions and determinism."""
    config = ExperimentConfig(random_seed=42)
    X, y, feature_names, target_names = load_and_validate_dataset(config)
    splits = split_dataset(X, y, feature_names, target_names, config)

    # Check partition dimensions (80% of 150 = 120, 20% of 150 = 30)
    assert splits.X_train.shape == (120, 4)
    assert splits.X_test.shape == (30, 4)
    assert splits.y_train.shape == (120,)
    assert splits.y_test.shape == (30,)

    # Verify strict class stratification (50 samples per class -> 40 train, 10 test)
    train_counts = splits.y_train.value_counts().to_dict()
    test_counts = splits.y_test.value_counts().to_dict()

    for cls in [0, 1, 2]:
        assert train_counts[cls] == 40
        assert test_counts[cls] == 10

    # Ensure no index overlap between train and test
    train_indices = set(splits.X_train.index)
    test_indices = set(splits.X_test.index)
    assert train_indices.isdisjoint(test_indices)
    assert len(train_indices.union(test_indices)) == 150

"""
Data loading, validation, and stratified splitting module.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ..config import ExperimentConfig, get_default_config


@dataclass
class DatasetSplits:
    """Dataclass holding strictly isolated train and holdout test partitions."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    target_names: list[str]
    raw_df: pd.DataFrame


def validate_iris_dataframe(
    df: pd.DataFrame,
    target_col: str = "target",
    expected_samples: int = 150,
    expected_features: int = 4,
    expected_classes: int = 3,
) -> None:
    """Validate data integrity and structural invariants of the dataset."""
    if df.shape[0] != expected_samples:
        raise ValueError(
            f"Data integrity error: Expected {expected_samples} rows, found {df.shape[0]}."
        )

    feature_cols = [c for c in df.columns if c != target_col]
    if len(feature_cols) != expected_features:
        raise ValueError(
            f"Data integrity error: Expected {expected_features} features, found {len(feature_cols)}."
        )

    if df.isnull().any().any():
        raise ValueError("Data integrity error: Dataset contains null/NaN values.")

    unique_classes = df[target_col].nunique()
    if unique_classes != expected_classes:
        raise ValueError(
            f"Data integrity error: Expected {expected_classes} classes, found {unique_classes}."
        )

    # Check for strictly positive morphological feature dimensions
    for col in feature_cols:
        if (df[col] <= 0).any():
            raise ValueError(
                f"Data integrity error: Feature '{col}' contains non-positive values."
            )


def load_and_validate_dataset(
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Load the benchmark Iris dataset, validate integrity, and return features and targets.
    Extensible for custom file inputs or alternative data sources.
    """
    cfg = config or get_default_config()

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    validate_iris_dataframe(
        df,
        target_col="target",
        expected_samples=cfg.data.expected_samples,
        expected_features=cfg.data.expected_features,
        expected_classes=cfg.data.expected_classes,
    )

    feature_names = list(iris.feature_names)
    target_names = [str(name) for name in iris.target_names]

    X = df[feature_names]
    y = df["target"]

    return X, y, feature_names, target_names


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    target_names: list[str],
    config: ExperimentConfig | None = None,
) -> DatasetSplits:
    """
    Perform a strict stratified train/test split.
    The test partition must remain isolated until final evaluation.
    """
    cfg = config or get_default_config()

    stratify_target = y if cfg.data.stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.random_seed,
        shuffle=True,
        stratify=stratify_target,
    )

    full_df = pd.concat([X, y], axis=1)

    return DatasetSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
        raw_df=full_df,
    )

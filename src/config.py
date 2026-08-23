"""
Configuration management module for the ML pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    test_size: float = 0.20
    stratify: bool = True
    expected_samples: int = 150
    expected_features: int = 4
    expected_classes: int = 3
    feature_names: list[str] = field(
        default_factory=lambda: [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
    )
    target_names: list[str] = field(
        default_factory=lambda: ["setosa", "versicolor", "virginica"]
    )


@dataclass
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    scoring_metric: str = "accuracy"
    refit_metric: str = "accuracy"


@dataclass
class PathsConfig:
    artifacts_dir: Path = Path("artifacts")
    metrics_dir: Path = Path("artifacts/metrics")
    experiments_dir: Path = Path("artifacts/experiments")
    predictions_dir: Path = Path("artifacts/predictions")
    plots_dir: Path = Path("artifacts/plots")
    models_dir: Path = Path("artifacts/models")

    def ensure_directories(self) -> None:
        """Create all artifact directories if they do not exist."""
        for path in [
            self.artifacts_dir,
            self.metrics_dir,
            self.experiments_dir,
            self.predictions_dir,
            self.plots_dir,
            self.models_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    name: str = "iris_multimodel_benchmark"
    random_seed: int = 42
    description: str = "Production-quality leakage-free multi-model benchmarking"
    data: DataConfig = field(default_factory=DataConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path | None = None) -> "ExperimentConfig":
        """Load configuration from a YAML file, or return default if missing."""
        if yaml_path is None or not Path(yaml_path).exists():
            config = cls()
            config.paths.ensure_directories()
            return config

        with open(yaml_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        exp_data = raw_data.get("experiment", {})
        data_cfg = raw_data.get("data", {})
        cv_cfg = raw_data.get("cv", {})
        paths_cfg = raw_data.get("paths", {})

        config = cls(
            name=exp_data.get("name", "iris_multimodel_benchmark"),
            random_seed=exp_data.get("random_seed", 42),
            description=exp_data.get(
                "description", "Production-quality leakage-free ML benchmark"
            ),
            data=DataConfig(
                test_size=data_cfg.get("test_size", 0.20),
                stratify=data_cfg.get("stratify", True),
                expected_samples=data_cfg.get("expected_samples", 150),
                expected_features=data_cfg.get("expected_features", 4),
                expected_classes=data_cfg.get("expected_classes", 3),
                feature_names=data_cfg.get(
                    "feature_names",
                    [
                        "sepal length (cm)",
                        "sepal width (cm)",
                        "petal length (cm)",
                        "petal width (cm)",
                    ],
                ),
                target_names=data_cfg.get(
                    "target_names", ["setosa", "versicolor", "virginica"]
                ),
            ),
            cv=CVConfig(
                n_splits=cv_cfg.get("n_splits", 5),
                shuffle=cv_cfg.get("shuffle", True),
                scoring_metric=cv_cfg.get("scoring_metric", "accuracy"),
                refit_metric=cv_cfg.get("refit_metric", "accuracy"),
            ),
            paths=PathsConfig(
                artifacts_dir=Path(paths_cfg.get("artifacts_dir", "artifacts")),
                metrics_dir=Path(paths_cfg.get("metrics_dir", "artifacts/metrics")),
                experiments_dir=Path(
                    paths_cfg.get("experiments_dir", "artifacts/experiments")
                ),
                predictions_dir=Path(
                    paths_cfg.get("predictions_dir", "artifacts/predictions")
                ),
                plots_dir=Path(paths_cfg.get("plots_dir", "artifacts/plots")),
                models_dir=Path(paths_cfg.get("models_dir", "artifacts/models")),
            ),
        )
        config.paths.ensure_directories()
        return config


def get_default_config() -> ExperimentConfig:
    """Helper to get experiment config searching standard locations."""
    config_file = Path("config/config.yaml")
    return ExperimentConfig.from_yaml(config_file if config_file.exists() else None)

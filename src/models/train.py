"""
Multi-model benchmarking and hyperparameter optimization module.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

# Suppress sklearn/future warnings during grid search
warnings.filterwarnings("ignore")

from ..config import ExperimentConfig, get_default_config
from .pipeline_factory import get_candidate_models_and_grids


@dataclass
class CVFoldMetrics:
    """Dataclass holding aggregate cross-validation statistics."""

    accuracy_mean: float
    accuracy_std: float
    precision_macro_mean: float
    precision_macro_std: float
    recall_macro_mean: float
    recall_macro_std: float
    f1_macro_mean: float
    f1_macro_std: float
    f1_weighted_mean: float
    f1_weighted_std: float


@dataclass
class BenchmarkResult:
    """Dataclass representing the evaluated outcome of an algorithm."""

    model_name: str
    best_params: dict[str, Any]
    best_estimator: Any
    cv_metrics: CVFoldMetrics
    grid_search_object: GridSearchCV
    cv_results_df: pd.DataFrame


class ModelBenchmarkEngine:
    """Engine responsible for running stratified cross-validated model tournament."""

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or get_default_config()
        self.cv = StratifiedKFold(
            n_splits=self.config.cv.n_splits,
            shuffle=self.config.cv.shuffle,
            random_state=self.config.random_seed,
        )

    def evaluate_candidate_model(
        self,
        model_name: str,
        pipeline: Any,
        param_grid: dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> BenchmarkResult:
        """Run GridSearchCV and detailed cross-validation for a candidate model."""
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.config.cv.scoring_metric,
            refit=True,
            n_jobs=None,
            return_train_score=True,
        )

        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_

        # Perform multi-metric cross-validation with best estimator
        scoring_dict = {
            "accuracy": "accuracy",
            "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
            "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
        }

        cv_scores = cross_validate(
            best_estimator,
            X_train,
            y_train,
            cv=self.cv,
            scoring=scoring_dict,
            n_jobs=None,
        )

        cv_metrics = CVFoldMetrics(
            accuracy_mean=float(np.mean(cv_scores["test_accuracy"])),
            accuracy_std=float(np.std(cv_scores["test_accuracy"])),
            precision_macro_mean=float(np.mean(cv_scores["test_precision_macro"])),
            precision_macro_std=float(np.std(cv_scores["test_precision_macro"])),
            recall_macro_mean=float(np.mean(cv_scores["test_recall_macro"])),
            recall_macro_std=float(np.std(cv_scores["test_recall_macro"])),
            f1_macro_mean=float(np.mean(cv_scores["test_f1_macro"])),
            f1_macro_std=float(np.std(cv_scores["test_f1_macro"])),
            f1_weighted_mean=float(np.mean(cv_scores["test_f1_weighted"])),
            f1_weighted_std=float(np.std(cv_scores["test_f1_weighted"])),
        )

        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cv_results_df["model_name"] = model_name

        return BenchmarkResult(
            model_name=model_name,
            best_params=grid_search.best_params_,
            best_estimator=best_estimator,
            cv_metrics=cv_metrics,
            grid_search_object=grid_search,
            cv_results_df=cv_results_df,
        )

    def run_benchmark(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        candidate_models: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, BenchmarkResult], pd.DataFrame, BenchmarkResult]:
        """
        Benchmark candidate models, rank them, and select the champion.
        Champion is chosen solely based on CV results on the training set.
        """
        candidates = candidate_models or get_candidate_models_and_grids(
            random_seed=self.config.random_seed
        )
        results: dict[str, BenchmarkResult] = {}
        comparison_rows: list[dict[str, Any]] = []
        all_cv_dfs: list[pd.DataFrame] = []

        for name, config_dict in candidates.items():
            result = self.evaluate_candidate_model(
                model_name=name,
                pipeline=config_dict["pipeline"],
                param_grid=config_dict["param_grid"],
                X_train=X_train,
                y_train=y_train,
            )
            results[name] = result
            all_cv_dfs.append(result.cv_results_df)

            comparison_rows.append(
                {
                    "Model": name,
                    "CV Accuracy Mean": result.cv_metrics.accuracy_mean,
                    "CV Accuracy Std": result.cv_metrics.accuracy_std,
                    "CV Precision Macro": result.cv_metrics.precision_macro_mean,
                    "CV Recall Macro": result.cv_metrics.recall_macro_mean,
                    "CV F1 Macro": result.cv_metrics.f1_macro_mean,
                    "CV F1 Weighted": result.cv_metrics.f1_weighted_mean,
                    "Best Hyperparameters": str(result.best_params),
                }
            )

        comparison_df = pd.DataFrame(comparison_rows)
        # Sort by CV Accuracy Mean descending, then lowest std, then CV F1 Macro
        comparison_df = comparison_df.sort_values(
            by=["CV Accuracy Mean", "CV F1 Macro", "CV Accuracy Std"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        # Champion model is the top-ranked model from CV training folds
        champion_name = comparison_df.iloc[0]["Model"]
        champion_result = results[champion_name]

        # Save artifacts
        self.config.paths.ensure_directories()
        comparison_path = self.config.paths.experiments_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)

        combined_cv_df = pd.concat(all_cv_dfs, ignore_index=True)
        cv_path = self.config.paths.experiments_dir / "cv_results.csv"
        combined_cv_df.to_csv(cv_path, index=False)

        return results, comparison_df, champion_result

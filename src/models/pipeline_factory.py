"""
Pipeline factory module for building atomic, leakage-free scikit-learn pipelines.
"""

from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_model_pipeline(
    model_name: str,
    estimator: Any,
    use_scaler: bool = True,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline ensuring preprocessing is strictly
    encapsulated with the estimator.
    """
    steps: list[tuple[str, Any]] = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    else:
        steps.append(("scaler", "passthrough"))

    steps.append(("classifier", estimator))
    return Pipeline(steps=steps)


def get_candidate_models_and_grids(
    random_seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Return candidate estimators, their pipeline scaling policy, and
    hyperparameter search spaces.
    """
    candidates = {
        "K-Nearest Neighbors": {
            "pipeline": get_model_pipeline(
                "KNN",
                KNeighborsClassifier(),
                use_scaler=True,
            ),
            "param_grid": {
                "classifier__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            },
        },
        "Logistic Regression": {
            "pipeline": get_model_pipeline(
                "Logistic Regression",
                LogisticRegression(random_state=random_seed, max_iter=1000),
                use_scaler=True,
            ),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0, 50.0, 100.0],
                "classifier__solver": ["lbfgs"],
            },
        },
        "Support Vector Machine": {
            "pipeline": get_model_pipeline(
                "SVM",
                SVC(random_state=random_seed, probability=True),
                use_scaler=True,
            ),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0, 100.0],
                "classifier__gamma": ["scale", "auto"],
                "classifier__kernel": ["linear", "rbf"],
            },
        },
        "Decision Tree": {
            "pipeline": get_model_pipeline(
                "Decision Tree",
                DecisionTreeClassifier(random_state=random_seed),
                use_scaler=False,
            ),
            "param_grid": {
                "classifier__max_depth": [None, 2, 3, 4, 5],
                "classifier__min_samples_split": [2, 5],
                "classifier__criterion": ["gini", "entropy"],
            },
        },
        "Random Forest": {
            "pipeline": get_model_pipeline(
                "Random Forest",
                RandomForestClassifier(random_state=random_seed),
                use_scaler=False,
            ),
            "param_grid": {
                "classifier__n_estimators": [25, 50, 100],
                "classifier__max_depth": [None, 3, 4, 5],
                "classifier__min_samples_split": [2, 5],
                "classifier__max_features": ["sqrt", None],
            },
        },
        "Gradient Boosting": {
            "pipeline": get_model_pipeline(
                "Gradient Boosting",
                GradientBoostingClassifier(random_state=random_seed),
                use_scaler=False,
            ),
            "param_grid": {
                "classifier__n_estimators": [25, 50, 100],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__max_depth": [2, 3],
            },
        },
        "HistGradientBoosting": {
            "pipeline": get_model_pipeline(
                "HistGradientBoosting",
                HistGradientBoostingClassifier(random_state=random_seed),
                use_scaler=False,
            ),
            "param_grid": {
                "classifier__max_iter": [25, 50, 100],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__max_depth": [2, 3],
            },
        },
    }
    return candidates


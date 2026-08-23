"""
Diagnostic visualization module for publishing and presentation-grade plots.
"""

from pathlib import Path
from typing import Any

import matplotlib

# Set non-interactive backend to ensure headless CI/terminal execution without GUI popups
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import ExperimentConfig, get_default_config
from ..data.loader import DatasetSplits
from ..models.evaluate import ErrorAnalysisResult


def plot_confusion_matrix(
    cm: list[list[int]],
    target_names: list[str],
    save_path: Path,
    title: str = "Champion Model: Confusion Matrix",
) -> None:
    """Render and save an annotated confusion matrix heatmap."""
    cm_arr = np.array(cm)
    plt.figure(figsize=(7, 6))

    # Calculate percentage annotations
    cm_sum = np.sum(cm_arr, axis=1, keepdims=True)
    cm_perc = np.divide(cm_arr, cm_sum, out=np.zeros_like(cm_arr, dtype=float), where=cm_sum != 0) * 100

    annot_matrix = np.empty_like(cm_arr, dtype=object)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            annot_matrix[i, j] = f"{cm_arr[i, j]}\n({cm_perc[i, j]:.1f}%)"

    sns.heatmap(
        cm_arr,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=False,
        linewidths=1.2,
        linecolor="#f0f4f8",
        annot_kws={"size": 11, "weight": "bold"},
    )
    plt.title(title, fontsize=13, weight="bold", pad=15)
    plt.xlabel("Predicted Species", fontsize=11, weight="semibold", labelpad=10)
    plt.ylabel("True Species", fontsize=11, weight="semibold", labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Render a ranked horizontal bar chart comparing cross-validated models."""
    plt.figure(figsize=(10, 6))

    df_sorted = comparison_df.sort_values(by="CV Accuracy Mean", ascending=True)
    models = df_sorted["Model"].tolist()
    means = (df_sorted["CV Accuracy Mean"] * 100).tolist()
    stds = (df_sorted["CV Accuracy Std"] * 100).tolist()

    # Color palette highlighting top model
    colors = ["#1a3a5c" if i == len(models) - 1 else "#5c7c9c" for i in range(len(models))]

    bars = plt.barh(models, means, xerr=stds, capsize=4, color=colors, edgecolor="#0f2338", alpha=0.9)
    plt.xlim(85, 101)
    plt.xlabel("Stratified 5-Fold CV Accuracy (%)", fontsize=11, weight="semibold")
    plt.title("Model Tournament Leaderboard (Mean CV Accuracy ± 1 Std)", fontsize=13, weight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Annotate mean values on bars
    for bar, mean_val, std_val in zip(bars, means, stds):
        plt.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_val:.2f}% (±{std_val:.2f})",
            va="center",
            ha="left",
            fontsize=9.5,
            weight="bold",
            color="#1a3a5c",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_performance(
    comparison_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Render grouped bar chart comparing multiple metrics across top algorithms."""
    plt.figure(figsize=(11, 6))

    metrics = ["CV Accuracy Mean", "CV Precision Macro", "CV Recall Macro", "CV F1 Macro", "CV F1 Weighted"]
    clean_metric_names = ["Accuracy", "Precision", "Recall", "F1 (Macro)", "F1 (Weighted)"]

    models = comparison_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.15

    palette = ["#1a3a5c", "#2a7a4c", "#e05c2a", "#8e44ad", "#d35400"]

    for i, (metric, clean_name, color) in enumerate(zip(metrics, clean_metric_names, palette)):
        values = [comparison_df.loc[comparison_df["Model"] == m, metric].values[0] for m in models]
        offset = (i - 2) * width
        plt.bar(x + offset, values, width, label=clean_name, color=color, alpha=0.85, edgecolor="#333333")

    plt.ylim(0.85, 1.02)
    plt.xticks(x, models, rotation=20, ha="right", fontsize=10, weight="semibold")
    plt.ylabel("Score (0.0 - 1.0)", fontsize=11, weight="semibold")
    plt.title("Multi-Metric Cross-Validation Benchmark Comparison", fontsize=13, weight="bold", pad=15)
    plt.legend(frameon=True, loc="lower right", fontsize=9.5)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_space_and_errors(
    splits: DatasetSplits,
    error_analysis: ErrorAnalysisResult,
    pred_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Render 2D feature space projections with training clusters, test instances,
    and prominent error callouts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"setosa": "#1a3a5c", "versicolor": "#e05c2a", "virginica": "#2a7a4c"}
    target_names = splits.target_names

    # Panel 1: Petal Length vs Petal Width
    for name in target_names:
        train_mask = [target_names[i] == name for i in splits.y_train.values]
        ax1.scatter(
            splits.X_train.loc[train_mask, "petal length (cm)"],
            splits.X_train.loc[train_mask, "petal width (cm)"],
            color=colors[name],
            alpha=0.35,
            s=40,
            label=f"{name} (train)",
        )

        test_correct = pred_df[(pred_df["actual_species"] == name) & (pred_df["is_correct"])]
        ax1.scatter(
            test_correct["petal length (cm)"],
            test_correct["petal width (cm)"],
            color=colors[name],
            edgecolor="#ffffff",
            linewidth=1.2,
            alpha=0.9,
            s=75,
            marker="o",
            label=f"{name} (test correct)",
        )

    # Highlight test errors on Panel 1
    test_errors = pred_df[~pred_df["is_correct"]]
    if not test_errors.empty:
        ax1.scatter(
            test_errors["petal length (cm)"],
            test_errors["petal width (cm)"],
            color="#ff0000",
            edgecolor="#000000",
            linewidth=1.8,
            s=160,
            marker="X",
            label="Misclassified Test Sample",
            zorder=10,
        )
        for _, err_row in test_errors.iterrows():
            ax1.annotate(
                f"True: {err_row['actual_species']}\nPred: {err_row['predicted_species']}\n(Conf: {err_row['confidence']:.2f})",
                xy=(err_row["petal length (cm)"], err_row["petal width (cm)"]),
                xytext=(err_row["petal length (cm)"] - 1.2, err_row["petal width (cm)"] + 0.3),
                arrowprops={"facecolor": "#ff0000", "shrink": 0.08, "width": 1.5, "headwidth": 6},
                fontsize=8.5,
                weight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff0f0", "edgecolor": "#ff0000"},
            )

    ax1.set_title("Feature Space: Petal Length vs Width", fontsize=12, weight="bold")
    ax1.set_xlabel("Petal Length (cm)", fontsize=10, weight="semibold")
    ax1.set_ylabel("Petal Width (cm)", fontsize=10, weight="semibold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: Sepal Length vs Sepal Width
    for name in target_names:
        train_mask = [target_names[i] == name for i in splits.y_train.values]
        ax2.scatter(
            splits.X_train.loc[train_mask, "sepal length (cm)"],
            splits.X_train.loc[train_mask, "sepal width (cm)"],
            color=colors[name],
            alpha=0.35,
            s=40,
        )

        test_correct = pred_df[(pred_df["actual_species"] == name) & (pred_df["is_correct"])]
        ax2.scatter(
            test_correct["sepal length (cm)"],
            test_correct["sepal width (cm)"],
            color=colors[name],
            edgecolor="#ffffff",
            linewidth=1.2,
            alpha=0.9,
            s=75,
            marker="o",
        )

    if not test_errors.empty:
        ax2.scatter(
            test_errors["sepal length (cm)"],
            test_errors["sepal width (cm)"],
            color="#ff0000",
            edgecolor="#000000",
            linewidth=1.8,
            s=160,
            marker="X",
            zorder=10,
        )

    ax2.set_title("Feature Space: Sepal Length vs Width", fontsize=12, weight="bold")
    ax2.set_xlabel("Sepal Length (cm)", fontsize=10, weight="semibold")
    ax2.set_ylabel("Sepal Width (cm)", fontsize=10, weight="semibold")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Iris Feature Distribution & Holdout Error Analysis", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_diagnostic_plots(
    comparison_df: pd.DataFrame,
    final_metrics: dict[str, Any],
    error_analysis: ErrorAnalysisResult,
    splits: DatasetSplits,
    pred_df: pd.DataFrame,
    config: ExperimentConfig | None = None,
) -> dict[str, Path]:
    """Generate and save the entire suite of diagnostic visual artifacts."""
    cfg = config or get_default_config()
    cfg.paths.ensure_directories()

    cm_path = cfg.paths.plots_dir / "confusion_matrix.png"
    comp_path = cfg.paths.plots_dir / "model_comparison.png"
    cv_path = cfg.paths.plots_dir / "cv_performance.png"
    feat_path = cfg.paths.plots_dir / "feature_space.png"

    plot_confusion_matrix(
        cm=final_metrics["confusion_matrix"],
        target_names=splits.target_names,
        save_path=cm_path,
        title=f"Champion Model ({final_metrics['champion_model']}) - Confusion Matrix",
    )

    plot_model_comparison(
        comparison_df=comparison_df,
        save_path=comp_path,
    )

    plot_cv_performance(
        comparison_df=comparison_df,
        save_path=cv_path,
    )

    plot_feature_space_and_errors(
        splits=splits,
        error_analysis=error_analysis,
        pred_df=pred_df,
        save_path=feat_path,
    )

    return {
        "confusion_matrix": cm_path,
        "model_comparison": comp_path,
        "cv_performance": cv_path,
        "feature_space": feat_path,
    }

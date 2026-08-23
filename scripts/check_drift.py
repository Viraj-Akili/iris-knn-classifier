"""
Offline Statistical Data Drift Analysis CLI.
Compares production feature distributions against the baseline training distribution.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import ExperimentConfig
from src.data.loader import load_and_validate_dataset, split_dataset
from src.monitoring.drift import DataDriftDetector


def generate_synthetic_drifted_batch(baseline_df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    """
    Generate a synthetic dataset with artificial distribution shifts (e.g. greenhouse fertilizer effect).
    Simulates increased petal lengths and altered sepal widths.
    """
    np.random.seed(99)
    drifted = pd.DataFrame()
    drifted["sepal length (cm)"] = baseline_df["sepal length (cm)"].sample(n_samples, replace=True).values + np.random.normal(0.8, 0.3, n_samples)
    drifted["sepal width (cm)"] = baseline_df["sepal width (cm)"].sample(n_samples, replace=True).values - np.random.normal(0.5, 0.2, n_samples)
    drifted["petal length (cm)"] = baseline_df["petal length (cm)"].sample(n_samples, replace=True).values + np.random.normal(1.5, 0.4, n_samples)
    drifted["petal width (cm)"] = baseline_df["petal width (cm)"].sample(n_samples, replace=True).values + np.random.normal(0.6, 0.2, n_samples)
    return drifted


def run_drift_analysis(simulate_drift: bool = False) -> None:
    """Execute statistical data drift comparison against the training baseline."""
    config = ExperimentConfig(random_seed=42)
    X, y, feature_names, target_names = load_and_validate_dataset(config)
    splits = split_dataset(X, y, feature_names, target_names, config)

    baseline_df = splits.X_train.copy()

    if simulate_drift:
        print("\n>>> Scenario: SIMULATED PRODUCTION DRIFT (Artificially Shifted Feature Dimensions)")
        current_df = generate_synthetic_drifted_batch(baseline_df, n_samples=120)
    else:
        print("\n>>> Scenario: HOLDOUT TEST BATCH (Standard Non-Drifted Validation Partition)")
        current_df = splits.X_test.copy()

    detector = DataDriftDetector(
        baseline_df=baseline_df,
        ks_alpha=0.05,
        psi_warning_threshold=0.10,
        psi_drift_threshold=0.20,
    )

    summary = detector.evaluate_drift(current_df)

    print("=" * 105)
    print("  STATISTICAL DATA DRIFT DETECTION REPORT")
    print(f"  Baseline Reference Samples : {len(baseline_df)} (Training Partition)")
    print(f"  Current Evaluation Samples : {len(current_df)}")
    print(f"  Significance Level (alpha) : {detector.ks_alpha}")
    print(f"  PSI Warning Threshold      : {detector.psi_warning_threshold} | Drift Threshold: {detector.psi_drift_threshold}")
    print("=" * 105)

    header = f"{'Feature':<22} | {'Baseline (Mean +/- Std)':<22} | {'Current (Mean +/- Std)':<22} | {'KS p-val':<9} | {'Wass (cm)':<10} | {'PSI':<7} | {'Status':<15}"
    print(header)
    print("-" * 110)

    for r in summary.feature_reports:
        base_stat = f"{r.baseline_mean:.2f} +/- {r.baseline_std:.2f}"
        curr_stat = f"{r.current_mean:.2f} +/- {r.current_std:.2f}"
        row_str = (
            f"{r.feature_name:<22} | "
            f"{base_stat:<22} | "
            f"{curr_stat:<22} | "
            f"{r.ks_p_value:<9.4f} | "
            f"{r.wasserstein_distance:<10.4f} | "
            f"{r.psi:<7.4f} | "
            f"{r.drift_status:<15}"
        )
        print(row_str)

    print("=" * 105)
    print(f"  Summary: {summary.drifted_features_count} drifted, {summary.warning_features_count} warnings / {summary.total_features_evaluated} features evaluated.")
    print(f"  Overall Dataset Drift Flag : {'[DRIFT DETECTED]' if summary.dataset_drift_detected else '[STABLE - NO DRIFT]'}")
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical ML Data Drift Inspector")
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Run drift detection against artificially shifted sample distributions.",
    )
    args = parser.parse_args()

    # Run standard validation partition check
    run_drift_analysis(simulate_drift=False)

    # Also display simulated drift scenario if requested or for full demonstration
    if args.simulate_drift:
        run_drift_analysis(simulate_drift=True)

"""
Statistical Data Drift Detection Module.
Evaluates distribution shifts between baseline training data and production inference batches.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def calculate_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    num_bins: int | None = None,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between baseline and current distributions.
    Uses adaptive quantile binning based on sample size.

    PSI interpretation:
    - PSI < 0.10: No significant shift / stable.
    - 0.10 <= PSI < 0.20: Moderate shift / warning.
    - PSI >= 0.20: Significant distribution drift.
    """
    baseline = np.asarray(baseline)
    current = np.asarray(current)

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    # Determine adaptive bin count to avoid empty bin instability on smaller evaluation batches
    if num_bins is None:
        effective_n = min(len(baseline), len(current))
        num_bins = min(10, max(4, effective_n // 6))

    # Determine quantile bin edges from baseline
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, quantiles)
    bin_edges = np.unique(bin_edges)  # Remove identical edges

    if len(bin_edges) < 2:
        # Fallback to linear range if quantiles collapse
        bin_edges = np.linspace(min(baseline.min(), current.min()), max(baseline.max(), current.max()), num_bins + 1)

    # Calculate frequency counts in each bin
    base_counts, _ = np.histogram(baseline, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to fractions with smoothing epsilon to prevent division by zero
    base_pct = (base_counts + epsilon) / (len(baseline) + epsilon * len(base_counts))
    curr_pct = (curr_counts + epsilon) / (len(current) + epsilon * len(curr_counts))

    # PSI formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_val = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(max(0.0, psi_val))


def calculate_ks_test(baseline: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """
    Two-Sample Kolmogorov-Smirnov test for continuous univariate distributions.
    Returns (ks_statistic, p_value).
    """
    res = ks_2samp(baseline, current)
    return float(res.statistic), float(res.pvalue)


def calculate_wasserstein(baseline: np.ndarray, current: np.ndarray) -> float:
    """
    Wasserstein Distance (Earth Mover's Distance).
    Measures physical distribution distance in original feature units (centimeters).
    """
    return float(wasserstein_distance(baseline, current))


@dataclass
class FeatureDriftReport:
    """Statistical drift report for a single continuous feature."""
    feature_name: str
    baseline_count: int
    current_count: int
    baseline_mean: float
    baseline_std: float
    current_mean: float
    current_std: float
    ks_statistic: float
    ks_p_value: float
    wasserstein_distance: float
    psi: float
    drift_status: str  # "NO_DRIFT", "WARNING", "DRIFT_DETECTED"
    reason: str


@dataclass
class DatasetDriftSummary:
    """Overall dataset drift report containing per-feature metrics."""
    total_features_evaluated: int
    drifted_features_count: int
    warning_features_count: int
    dataset_drift_detected: bool
    feature_reports: list[FeatureDriftReport] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Format feature reports into a clean tabular DataFrame."""
        rows = []
        for r in self.feature_reports:
            rows.append({
                "Feature": r.feature_name,
                "Baseline (Mean ± Std)": f"{r.baseline_mean:.2f} ± {r.baseline_std:.2f}",
                "Current (Mean ± Std)": f"{r.current_mean:.2f} ± {r.current_std:.2f}",
                "KS Stat (D)": round(r.ks_statistic, 4),
                "KS p-val": round(r.ks_p_value, 4),
                "Wasserstein (cm)": round(r.wasserstein_distance, 4),
                "PSI": round(r.psi, 4),
                "Status": r.drift_status,
            })
        return pd.DataFrame(rows)


class DataDriftDetector:
    """
    Production statistical data drift detector.
    Compares current feature distributions against a baseline training distribution.
    """

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        ks_alpha: float = 0.05,
        psi_warning_threshold: float = 0.10,
        psi_drift_threshold: float = 0.20,
    ) -> None:
        self.baseline_df = baseline_df.copy()
        self.ks_alpha = ks_alpha
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_drift_threshold = psi_drift_threshold

    def evaluate_drift(
        self,
        current_df: pd.DataFrame,
        features: list[str] | None = None,
    ) -> DatasetDriftSummary:
        """
        Evaluate distribution shift between baseline and current data batches.
        """
        eval_features = features or [
            col for col in self.baseline_df.columns if col in current_df.columns
        ]

        feature_reports: list[FeatureDriftReport] = []
        drift_count = 0
        warning_count = 0

        for feat in eval_features:
            base_vals = self.baseline_df[feat].dropna().values
            curr_vals = current_df[feat].dropna().values

            if len(base_vals) < 5 or len(curr_vals) < 5:
                # Insufficient sample size for reliable hypothesis testing
                report = FeatureDriftReport(
                    feature_name=feat,
                    baseline_count=len(base_vals),
                    current_count=len(curr_vals),
                    baseline_mean=float(np.mean(base_vals)) if len(base_vals) > 0 else 0.0,
                    baseline_std=float(np.std(base_vals)) if len(base_vals) > 0 else 0.0,
                    current_mean=float(np.mean(curr_vals)) if len(curr_vals) > 0 else 0.0,
                    current_std=float(np.std(curr_vals)) if len(curr_vals) > 0 else 0.0,
                    ks_statistic=0.0,
                    ks_p_value=1.0,
                    wasserstein_distance=0.0,
                    psi=0.0,
                    drift_status="INSUFFICIENT_DATA",
                    reason="Sample size too small (<5 observations) for statistical testing.",
                )
                feature_reports.append(report)
                continue

            ks_stat, ks_pval = calculate_ks_test(base_vals, curr_vals)
            wass_dist = calculate_wasserstein(base_vals, curr_vals)
            psi_val = calculate_psi(base_vals, curr_vals)

            # Determine composite drift status
            if psi_val >= self.psi_drift_threshold and ks_pval < self.ks_alpha:
                status = "DRIFT_DETECTED"
                reason = f"High PSI ({psi_val:.3f} >= {self.psi_drift_threshold}) and statistically significant KS test (p={ks_pval:.4f} < {self.ks_alpha})."
                drift_count += 1
            elif psi_val >= self.psi_warning_threshold or ks_pval < self.ks_alpha:
                status = "WARNING"
                reason = f"Moderate distribution shift (PSI={psi_val:.3f}, KS p={ks_pval:.4f})."
                warning_count += 1
            else:
                status = "NO_DRIFT"
                reason = f"Distribution is stable (PSI={psi_val:.3f} < {self.psi_warning_threshold}, KS p={ks_pval:.4f} >= {self.ks_alpha})."

            report = FeatureDriftReport(
                feature_name=feat,
                baseline_count=len(base_vals),
                current_count=len(curr_vals),
                baseline_mean=float(np.mean(base_vals)),
                baseline_std=float(np.std(base_vals)),
                current_mean=float(np.mean(curr_vals)),
                current_std=float(np.std(curr_vals)),
                ks_statistic=ks_stat,
                ks_p_value=ks_pval,
                wasserstein_distance=wass_dist,
                psi=psi_val,
                drift_status=status,
                reason=reason,
            )
            feature_reports.append(report)

        dataset_drift = drift_count > 0 or warning_count >= max(1, len(eval_features) // 2)

        return DatasetDriftSummary(
            total_features_evaluated=len(eval_features),
            drifted_features_count=drift_count,
            warning_features_count=warning_count,
            dataset_drift_detected=dataset_drift,
            feature_reports=feature_reports,
        )

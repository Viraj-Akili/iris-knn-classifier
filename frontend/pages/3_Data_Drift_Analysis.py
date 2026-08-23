"""
Statistical Data Drift Analysis Interface.
Executes Two-Sample KS-test, Wasserstein distance, and PSI between baseline and evaluation datasets.
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Data Drift Analysis - Iris ML Console",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import IrisApiClient
from frontend.components.header import render_header, render_sidebar_connection
from frontend.utils.formatting import apply_custom_theme
from src.config import ExperimentConfig
from src.data.loader import load_and_validate_dataset, split_dataset
from src.monitoring.drift import DataDriftDetector

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_connection(api_client)
render_header(api_client)

st.markdown("## 🧪 Statistical Data Drift & Distribution Shift Inspector")
st.markdown(
    """
    Evaluate covariate shifts between the baseline training distribution ($N=120$)
    and evaluation batches using non-parametric hypothesis testing, geometric distance metrics,
    and population stability indexes.
    """
)

# Load baseline data
config = ExperimentConfig(random_seed=42)
X, y, feature_names, target_names = load_and_validate_dataset(config)
splits = split_dataset(X, y, feature_names, target_names, config)
baseline_df = splits.X_train.copy()

st.markdown("### ⚙️ Drift Evaluation Scenario")

col_scenario, col_params = st.columns([2, 1])

with col_scenario:
    scenario = st.radio(
        "Select Current Evaluation Batch:",
        options=[
            "1. Holdout Validation Partition (N=30, Expected Stable)",
            "2. Simulated Greenhouse Shift (N=120, Shifted Petal & Sepal Dimensions)",
        ],
        index=0,
    )

with col_params:
    alpha = st.slider("Significance Level (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    psi_thresh = st.slider("PSI Drift Threshold", min_value=0.10, max_value=0.30, value=0.20, step=0.05)

if "Holdout" in scenario:
    current_df = splits.X_test.copy()
    scenario_desc = "Holdout Test Set (30 samples drawn from same population as training folds)."
else:
    np.random.seed(99)
    current_df = pd.DataFrame()
    current_df["sepal length (cm)"] = baseline_df["sepal length (cm)"].sample(120, replace=True).values + np.random.normal(0.8, 0.3, 120)
    current_df["sepal width (cm)"] = baseline_df["sepal width (cm)"].sample(120, replace=True).values - np.random.normal(0.5, 0.2, 120)
    current_df["petal length (cm)"] = baseline_df["petal length (cm)"].sample(120, replace=True).values + np.random.normal(1.5, 0.4, 120)
    current_df["petal width (cm)"] = baseline_df["petal width (cm)"].sample(120, replace=True).values + np.random.normal(0.6, 0.2, 120)
    scenario_desc = "Synthetic dataset simulating environmental growth acceleration (increased petal size, narrowed sepals)."

detector = DataDriftDetector(
    baseline_df=baseline_df,
    ks_alpha=alpha,
    psi_warning_threshold=0.10,
    psi_drift_threshold=psi_thresh,
)

summary = detector.evaluate_drift(current_df)

st.markdown("---")
st.markdown("### 📋 Statistical Drift Analysis Report")
st.caption(scenario_desc)

# Summary metrics row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Reference Baseline Samples", f"{len(baseline_df)}")
with c2:
    st.metric("Evaluation Batch Samples", f"{len(current_df)}")
with c3:
    st.metric("Drifted Features Detected", f"{summary.drifted_features_count} / {summary.total_features_evaluated}")
with c4:
    if summary.dataset_drift_detected:
        st.markdown(
            """
            <div style="background:rgba(239,68,68,0.2); border:1px solid #ef4444; border-radius:8px; padding:10px; text-align:center;">
                <strong style="color:#ef4444; font-size:1.1rem;">🚨 DRIFT DETECTED</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:rgba(16,185,129,0.2); border:1px solid #10b981; border-radius:8px; padding:10px; text-align:center;">
                <strong style="color:#10b981; font-size:1.1rem;">✅ POPULATION STABLE</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Detailed results table
table_df = summary.to_dataframe()
st.dataframe(table_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 🧠 Statistical Methodology & Rationale")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        """
        **1. Two-Sample Kolmogorov-Smirnov (KS) Test**
        - Non-parametric comparison of continuous empirical cumulative distribution functions ($F_1(x)$ vs $F_2(x)$).
        - Sensitive to distribution shifts, spread, skewness, and tail divergence without binning artifacts.
        - Null Hypothesis ($H_0$): Distributions are identical. Flagged when $p\text{-value} < \\alpha$.
        """
    )

with m2:
    st.markdown(
        """
        **2. Wasserstein Distance (Earth Mover's Distance)**
        - Measures minimal geometric "work" to morph baseline into current distribution.
        - Preserves physical interpretability in **original feature units (centimeters)**.
        - Provides continuous magnitude regardless of sample size scaling.
        """
    )

with m3:
    st.markdown(
        """
        **3. Population Stability Index (PSI)**
        - Industry standard metric evaluating bin-wise population proportions:
          $$\\text{PSI} = \\sum (A_i - E_i) \\ln(A_i / E_i)$$
        - $\\text{PSI} < 0.10$: Stable.
        - $0.10 \\le \\text{PSI} < 0.20$: Moderate Shift.
        - $\\text{PSI} \\ge 0.20$: Significant Drift.
        """
    )

st.warning(
    "⚠️ **Sample Size Caution**: On small sample sizes ($N < 50$), quantile-based PSI can exhibit sensitivity to bin boundaries. "
    "Combining non-parametric KS-tests with Wasserstein distance provides the most reliable multi-dimensional signal."
)

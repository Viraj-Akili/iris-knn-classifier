"""
IRIS ML - Drift Module
Statistical feature drift monitoring (Two-Sample KS, Wasserstein distance, PSI).
Strictly uses the canonical stratified training baseline distribution (N=120).
"""

import logging

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Drift - IRIS ML",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import IrisApiClient
from frontend.components.header import render_header
from frontend.components.navigation import render_sidebar_navigation
from frontend.utils.formatting import apply_custom_theme
from src.config import get_default_config
from src.data.loader import load_and_validate_dataset, split_dataset
from src.monitoring.drift import DataDriftDetector

logger = logging.getLogger(__name__)

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_navigation(api_client)
render_header(
    api_client,
    title="Feature drift",
    subtitle="Compare current observations against the training reference distribution.",
)


@st.cache_data
def get_training_reference():
    """Load canonical training baseline using the verified ML pipeline configuration."""
    config = get_default_config()
    X, y, feature_names, target_names = load_and_validate_dataset(config)
    splits = split_dataset(X, y, feature_names, target_names, config)
    return splits.X_train, splits.X_test


reference_loaded = False
X_train, X_test = None, None

try:
    X_train, X_test = get_training_reference()
    reference_loaded = True
except (ValueError, FileNotFoundError, KeyError) as e:
    logger.error("Failed to load canonical reference dataset: %s", e)
    st.error("Reference dataset unavailable")
    st.caption("Drift analysis cannot be computed until the training reference distribution is available.")
except Exception as e:
    logger.exception("Unexpected error while loading reference dataset: %s", e)
    st.error("Reference dataset unavailable")
    st.caption("Drift analysis cannot be computed until the training reference distribution is available.")

if reference_loaded and X_train is not None and X_test is not None:
    detector = DataDriftDetector(baseline_df=X_train)

    # Comparison Dataset Selector
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        dataset_mode = st.selectbox(
            "Evaluation batch",
            options=[
                "Holdout Test Baseline (N=30, Unshifted)",
                "Simulated Production Shift (N=50, Morphological Drift)",
            ],
            index=0,
        )

    if "Unshifted" in dataset_mode:
        current_df = X_test
    else:
        np.random.seed(42)
        current_df = X_test.copy()
        current_df["sepal length (cm)"] = current_df["sepal length (cm)"] + np.random.normal(
            1.1, 0.3, len(current_df)
        )
        current_df["petal length (cm)"] = current_df["petal length (cm)"] + np.random.normal(
            1.6, 0.4, len(current_df)
        )

    # Reference vs Current Counts
    st.markdown(
        f"""
        <div class="status-bar" style="margin-top: 6px;">
            <div class="status-bar-item">Reference Baseline: <span style="color: #f8fafc; margin-left: 4px;">{len(X_train)} training observations</span></div>
            <span style="color: #334155;">|</span>
            <div class="status-bar-item">Current Batch: <span style="color: #f8fafc; margin-left: 4px;">{len(current_df)} observations</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(current_df) < 5:
        st.info("Insufficient observations for drift analysis (minimum 5 required).")
    else:
        # Compute real statistical drift
        summary = detector.evaluate_drift(current_df)

        # Format display table with requested columns
        table_rows = []
        for r in summary.feature_reports:
            table_rows.append({
                "Feature": r.feature_name.title(),
                "PSI": f"{r.psi:.3f}",
                "KS": f"{r.ks_statistic:.3f}",
                "Wasserstein": f"{r.wasserstein_distance:.3f}",
                "Status": r.drift_status,
            })

        st.markdown("### Drift metrics")
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.markdown("<hr style='margin: 14px 0;'/>", unsafe_allow_html=True)

        # Distribution Chart
        st.markdown("### Distribution comparison")
        feature_to_plot = st.selectbox(
            "Feature",
            options=list(X_train.columns),
            index=2,
        )

        ref_series = X_train[feature_to_plot]
        curr_series = current_df[feature_to_plot]

        df_plot = pd.DataFrame({
            "Value": list(ref_series) + list(curr_series),
            "Distribution": ["Reference (Training)" for _ in range(len(ref_series))]
            + ["Current Batch" for _ in range(len(curr_series))],
        })

        chart = (
            alt.Chart(df_plot)
            .transform_density(
                "Value",
                as_=["Value", "Density"],
                groupby=["Distribution"],
            )
            .mark_area(opacity=0.4)
            .encode(
                x=alt.X(
                    "Value:Q",
                    title=f"{feature_to_plot.title()} (cm)",
                    axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b"),
                ),
                y=alt.Y(
                    "Density:Q",
                    title="Density",
                    axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b"),
                ),
                color=alt.Color(
                    "Distribution:N",
                    scale=alt.Scale(
                        domain=["Reference (Training)", "Current Batch"],
                        range=["#3b82f6", "#f59e0b"],
                    ),
                    legend=alt.Legend(orient="top", title=None, labelColor="#f8fafc"),
                ),
            )
            .properties(height=220)
        )

        st.altair_chart(chart, use_container_width=True)

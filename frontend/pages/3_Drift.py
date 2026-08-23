"""
Drift Module for IRIS ML PLATFORM.
Statistical data drift monitoring (Two-Sample KS Test, Wasserstein distance, PSI).
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Drift - IRIS ML PLATFORM",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import IrisApiClient
from frontend.components.header import render_header, render_sidebar_connection
from frontend.utils.formatting import apply_custom_theme
from src.config import get_default_config
from src.data.loader import load_and_validate_dataset, split_dataset
from src.monitoring.drift import DataDriftDetector

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_connection(api_client)
render_header(api_client)

st.markdown("## Feature drift monitoring")
st.caption("Compare production observations against the training reference distribution.")

# Load reference training baseline
@st.cache_data
def get_reference_data():
    config = get_default_config()
    df, _, _ = load_and_validate_dataset(config.dataset_path)
    splits = split_dataset(df, config.test_size, config.random_state)
    return splits.X_train, splits.X_test

X_train, X_test = get_reference_data()
detector = DataDriftDetector(reference_data=X_train)

# Evaluation dataset selection
st.markdown("### Evaluation Dataset")
dataset_mode = st.radio(
    "Select comparison batch:",
    options=["Holdout Test Set (Unshifted baseline, N=30)", "Simulated Greenhouse Shift (Perturbed, N=50)"],
    index=0,
    horizontal=True,
)

if "Unshifted" in dataset_mode:
    current_df = X_test
else:
    np.random.seed(42)
    current_df = X_test.copy()
    current_df["sepal length (cm)"] = current_df["sepal length (cm)"] + np.random.normal(1.2, 0.3, len(current_df))
    current_df["petal length (cm)"] = current_df["petal length (cm)"] + np.random.normal(1.8, 0.4, len(current_df))

# Compute drift
drift_results = detector.detect_drift(current_df)

# Professional Summary Table
table_rows = []
for feat, metrics in drift_results.items():
    ks_stat = metrics.get("ks_statistic", 0.0)
    w_dist = metrics.get("wasserstein_distance", 0.0)
    psi_val = metrics.get("psi", 0.0)
    status_label = metrics.get("status", "STABLE")

    table_rows.append({
        "Feature": feat.title(),
        "PSI": f"{psi_val:.3f}",
        "KS Statistic": f"{ks_stat:.3f}",
        "Wasserstein Distance (cm)": f"{w_dist:.3f}",
        "Status": status_label,
    })

st.markdown("### Drift metrics summary")
st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# Feature Distribution Comparison
st.markdown("### Distribution comparison")
feature_to_plot = st.selectbox(
    "Select feature to inspect:",
    options=list(X_train.columns),
    index=2,
)

ref_series = X_train[feature_to_plot]
curr_series = current_df[feature_to_plot]

df_plot = pd.DataFrame({
    "Value": list(ref_series) + list(curr_series),
    "Distribution": ["Reference (Training)" for _ in range(len(ref_series))] + ["Production / Comparison" for _ in range(len(curr_series))],
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
        x=alt.X("Value:Q", title=f"{feature_to_plot.title()} (cm)", axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b")),
        y=alt.Y("Density:Q", title="Probability Density", axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b")),
        color=alt.Color(
            "Distribution:N",
            scale=alt.Scale(domain=["Reference (Training)", "Production / Comparison"], range=["#3b82f6", "#f59e0b"]),
            legend=alt.Legend(orient="top", title=None, labelColor="#f8fafc"),
        ),
    )
    .properties(height=260)
)

st.altair_chart(chart, use_container_width=True)

st.markdown(
    """
    <div style="font-size: 0.78rem; color: #64748b; margin-top: 8px;">
        Statistical methodology: Two-Sample Kolmogorov-Smirnov test evaluated at α = 0.05. Wasserstein distance represents the Earth Mover's Distance in physical measurement units (cm). Population Stability Index (PSI) measures empirical quantile shift against training bins.
    </div>
    """,
    unsafe_allow_html=True,
)

"""
Models Module for IRIS ML PLATFORM.
Multi-model cross-validation tournament benchmark and champion architecture ranking.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Models - IRIS ML PLATFORM",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import IrisApiClient
from frontend.components.header import render_header, render_sidebar_connection
from frontend.components.metrics_cards import render_kpi_card
from frontend.utils.formatting import apply_custom_theme

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_connection(api_client)
render_header(api_client)

st.markdown("## Model comparison")
st.caption("5-Fold Stratified Cross-Validation benchmark across 7 candidate algorithms on the 120-sample training partition.")

# Champion Summary Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi_card("Champion", "Support Vector Machine", "Selected algorithm")
with c2:
    render_kpi_card("CV Accuracy", "97.50% ± 2.04%", "Fold mean & standard deviation")
with c3:
    render_kpi_card("Macro F1", "0.9749", "Balanced class score")
with c4:
    render_kpi_card("Holdout Accuracy", "93.33%", "28/30 untouched test samples")

st.markdown("<br>", unsafe_allow_html=True)

# 7-Model Benchmark Table
BENCHMARK_DATA = [
    {
        "Model": "Support Vector Machine (Champion)",
        "CV Accuracy": "97.50%",
        "CV Std": "±2.04%",
        "Macro F1": "0.9749",
        "Weighted F1": "0.9749",
        "Holdout Accuracy": "93.33%",
        "Hyperparameters": "C=0.1, kernel='linear'",
    },
    {
        "Model": "K-Nearest Neighbors",
        "CV Accuracy": "96.67%",
        "CV Std": "±1.67%",
        "Macro F1": "0.9665",
        "Weighted F1": "0.9665",
        "Holdout Accuracy": "93.33%",
        "Hyperparameters": "n_neighbors=3, weights='uniform'",
    },
    {
        "Model": "Logistic Regression",
        "CV Accuracy": "96.67%",
        "CV Std": "±3.12%",
        "Macro F1": "0.9663",
        "Weighted F1": "0.9663",
        "Holdout Accuracy": "93.33%",
        "Hyperparameters": "C=10.0, solver='lbfgs'",
    },
    {
        "Model": "Random Forest",
        "CV Accuracy": "96.67%",
        "CV Std": "±3.12%",
        "Macro F1": "0.9663",
        "Weighted F1": "0.9663",
        "Holdout Accuracy": "93.33%",
        "Hyperparameters": "n_estimators=50, min_samples_split=5",
    },
    {
        "Model": "Decision Tree",
        "CV Accuracy": "95.83%",
        "CV Std": "±2.64%",
        "Macro F1": "0.9580",
        "Weighted F1": "0.9580",
        "Holdout Accuracy": "93.33%",
        "Hyperparameters": "criterion='gini', min_samples_split=5",
    },
    {
        "Model": "Gradient Boosting",
        "CV Accuracy": "95.83%",
        "CV Std": "±2.64%",
        "Macro F1": "0.9580",
        "Weighted F1": "0.9580",
        "Holdout Accuracy": "93.33%",
        "Hyperparameters": "learning_rate=0.05, n_estimators=25",
    },
    {
        "Model": "HistGradientBoosting",
        "CV Accuracy": "95.83%",
        "CV Std": "±2.64%",
        "Macro F1": "0.9580",
        "Weighted F1": "0.9580",
        "Holdout Accuracy": "93.33%",
        "Hyperparameters": "learning_rate=0.05, max_iter=25",
    },
]

st.markdown("### Benchmark leaderboard")
df_benchmark = pd.DataFrame(BENCHMARK_DATA)
st.dataframe(df_benchmark, use_container_width=True, hide_index=True)

st.markdown("---")

# Visual Diagnostic Figures
st.markdown("### Diagnostic plots")
col_img1, col_img2 = st.columns(2)

artifacts_dir = Path("artifacts/plots")
plot1 = artifacts_dir / "model_comparison.png"
plot2 = artifacts_dir / "cv_performance.png"

with col_img1:
    if plot1.exists():
        st.image(str(plot1), caption="Multi-Model Cross-Validation Metric Comparison", use_container_width=True)

with col_img2:
    if plot2.exists():
        st.image(str(plot2), caption="Fold-Level Accuracy Variance Across Candidates", use_container_width=True)

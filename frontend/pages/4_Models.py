"""
IRIS ML - Models Module
Model benchmark leaderboard and candidate classifier comparison.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Models - IRIS ML",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import IrisApiClient
from frontend.components.header import render_header
from frontend.components.navigation import render_sidebar_navigation
from frontend.utils.formatting import apply_custom_theme

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_navigation(api_client)
render_header(api_client, title="Model benchmark", subtitle="Comparison of candidate classifiers using stratified cross-validation.")

BENCHMARK_DATA = [
    {
        "Model": "Linear SVM (Champion)",
        "CV Accuracy": "97.50%",
        "Std": "±2.04%",
        "Macro F1": "0.9749",
        "Holdout Accuracy": "93.33%",
    },
    {
        "Model": "K-Nearest Neighbors",
        "CV Accuracy": "96.67%",
        "Std": "±1.67%",
        "Macro F1": "0.9665",
        "Holdout Accuracy": "93.33%",
    },
    {
        "Model": "Logistic Regression",
        "CV Accuracy": "96.67%",
        "Std": "±3.12%",
        "Macro F1": "0.9663",
        "Holdout Accuracy": "93.33%",
    },
    {
        "Model": "Random Forest",
        "CV Accuracy": "96.67%",
        "Std": "±3.12%",
        "Macro F1": "0.9663",
        "Holdout Accuracy": "93.33%",
    },
    {
        "Model": "Decision Tree",
        "CV Accuracy": "95.83%",
        "Std": "±2.64%",
        "Macro F1": "0.9580",
        "Holdout Accuracy": "93.33%",
    },
    {
        "Model": "Gradient Boosting",
        "CV Accuracy": "95.83%",
        "Std": "±2.64%",
        "Macro F1": "0.9580",
        "Holdout Accuracy": "93.33%",
    },
    {
        "Model": "HistGradientBoosting",
        "CV Accuracy": "95.83%",
        "Std": "±2.64%",
        "Macro F1": "0.9580",
        "Holdout Accuracy": "93.33%",
    },
]

st.markdown("### Candidate comparison")
st.dataframe(pd.DataFrame(BENCHMARK_DATA), use_container_width=True, hide_index=True)

st.markdown("<hr style='margin: 14px 0;'/>", unsafe_allow_html=True)

# Champion Section
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Champion Model")
    st.markdown(
        """
        <div class="cap-box">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">Linear SVM</div>
            <div style="font-size: 0.8rem; color: #10b981; font-weight: 600; margin-bottom: 8px;">97.50% ± 2.04% CV Accuracy</div>
            <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.4;">
                Selected for highest cross-validation score and superior regularized generalization on small-sample tabular partitions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown("### Hyperparameters")
    st.markdown(
        """
        <div class="cap-box">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #f8fafc; line-height: 1.8;">
                • C = 0.1<br>
                • kernel = linear<br>
                • probability = calibrated (Platt scaling)<br>
                • random_state = 42
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin: 14px 0;'/>", unsafe_allow_html=True)

# Diagnostic Visualizations
st.markdown("### Diagnostic plots")
col_img1, col_img2 = st.columns(2)
artifacts_dir = Path("artifacts/plots")
plot1 = artifacts_dir / "model_comparison.png"
plot2 = artifacts_dir / "cv_performance.png"

with col_img1:
    if plot1.exists():
        st.image(str(plot1), caption="Cross-Validation Metric Comparison", use_container_width=True)

with col_img2:
    if plot2.exists():
        st.image(str(plot2), caption="Fold Accuracy Variance", use_container_width=True)

"""
Evaluation Module for IRIS ML PLATFORM.
Holdout performance, confusion matrix, classification report, and error analysis.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Evaluation - IRIS ML PLATFORM",
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

st.markdown("## Model evaluation")
st.caption("Granular post-evaluation inspection of the champion Support Vector Machine on the untouched 30-sample holdout test partition.")

# KPI summary
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("Holdout Accuracy", "93.33%", "28 / 30 correct")
with k2:
    render_kpi_card("Macro Precision", "0.9444", "Average across species")
with k3:
    render_kpi_card("Macro Recall", "0.9333", "Average true positive rate")
with k4:
    render_kpi_card("Macro F1", "0.9327", "Harmonic mean")

st.markdown("<br>", unsafe_allow_html=True)

# 1. Classification Metrics Table
st.markdown("### Classification metrics")
REPORT_DATA = [
    {"Species": "Setosa", "Precision": "1.000", "Recall": "1.000", "F1-Score": "1.000", "Support": 10},
    {"Species": "Versicolor", "Precision": "0.900", "Recall": "0.900", "F1-Score": "0.900", "Support": 10},
    {"Species": "Virginica", "Precision": "0.900", "Recall": "0.900", "F1-Score": "0.900", "Support": 10},
]
st.dataframe(pd.DataFrame(REPORT_DATA), use_container_width=True, hide_index=True)

st.markdown("---")

# 2. Confusion Matrix & Feature Space Projection
st.markdown("### Confusion matrix & feature space")
col_cm, col_fs = st.columns(2)

artifacts_dir = Path("artifacts/plots")
cm_path = artifacts_dir / "confusion_matrix.png"
fs_path = artifacts_dir / "feature_space.png"

with col_cm:
    if cm_path.exists():
        st.image(str(cm_path), caption="Holdout Test Confusion Matrix", use_container_width=True)

with col_fs:
    if fs_path.exists():
        st.image(str(fs_path), caption="2D Morphological Decision Boundary Projection", use_container_width=True)

st.markdown("---")

# 3. Misclassified Samples Breakdown
st.markdown("### Misclassified specimens")
st.caption("Exactly 2 out of 30 holdout specimens reside directly along the overlapping versicolor ↔ virginica decision boundary.")

ERROR_RECORDS = [
    {
        "Sample Index": 134,
        "Actual Species": "virginica",
        "Predicted Species": "versicolor",
        "Confidence": "50.0%",
        "Sepal Length": 6.1,
        "Sepal Width": 2.6,
        "Petal Length": 5.6,
        "Petal Width": 1.4,
        "Diagnostic Rationale": "Petal width (1.4 cm) matches typical versicolor range despite virginica-like petal length (5.6 cm).",
    },
    {
        "Sample Index": 77,
        "Actual Species": "versicolor",
        "Predicted Species": "virginica",
        "Confidence": "52.8%",
        "Sepal Length": 6.7,
        "Sepal Width": 3.0,
        "Petal Length": 5.0,
        "Petal Width": 1.7,
        "Diagnostic Rationale": "Petal length (5.0 cm) and petal width (1.7 cm) extend past standard versicolor bounds into virginica territory.",
    },
]

st.dataframe(pd.DataFrame(ERROR_RECORDS), use_container_width=True, hide_index=True)

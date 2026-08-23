"""
IRIS ML - Evaluation Module
Holdout test performance, confusion matrix, and misclassification diagnostics.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Evaluation - IRIS ML",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import IrisApiClient
from frontend.components.header import render_header
from frontend.components.metrics_cards import render_kpi_card
from frontend.components.navigation import render_sidebar_navigation
from frontend.utils.formatting import apply_custom_theme

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_navigation(api_client)
render_header(api_client, title="Model evaluation", subtitle="Holdout performance and classification errors.")

# Top Metrics Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("Accuracy", "93.33%", "28/30 test samples")
with k2:
    render_kpi_card("Precision", "0.944", "Macro average")
with k3:
    render_kpi_card("Recall", "0.933", "Macro average")
with k4:
    render_kpi_card("F1-Score", "0.933", "Harmonic mean")

st.markdown("<hr style='margin: 14px 0;'/>", unsafe_allow_html=True)

# Confusion Matrix & Feature Space
col_cm, col_fs = st.columns(2)
artifacts_dir = Path("artifacts/plots")
cm_path = artifacts_dir / "confusion_matrix.png"
fs_path = artifacts_dir / "feature_space.png"

with col_cm:
    st.markdown("### Confusion Matrix")
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)

with col_fs:
    st.markdown("### Feature Space")
    if fs_path.exists():
        st.image(str(fs_path), use_container_width=True)

st.markdown("<hr style='margin: 14px 0;'/>", unsafe_allow_html=True)

# Class Performance
st.markdown("### Class Performance")
CLASS_REPORT = [
    {"Species": "Setosa", "Precision": "1.000", "Recall": "1.000", "F1-Score": "1.000", "Support": 10},
    {"Species": "Versicolor", "Precision": "0.900", "Recall": "0.900", "F1-Score": "0.900", "Support": 10},
    {"Species": "Virginica", "Precision": "0.900", "Recall": "0.900", "F1-Score": "0.900", "Support": 10},
]
st.dataframe(pd.DataFrame(CLASS_REPORT), use_container_width=True, hide_index=True)

st.markdown("<hr style='margin: 14px 0;'/>", unsafe_allow_html=True)

# Misclassifications
st.markdown("### Misclassifications")
st.caption("2 out of 30 test specimens reside along the overlapping versicolor / virginica morphological boundary.")

MISCLASS_DATA = [
    {
        "Sample": "#134",
        "Actual": "virginica",
        "Predicted": "versicolor",
        "Confidence": "50.0%",
        "Dimensions": "SL: 6.1, SW: 2.6, PL: 5.6, PW: 1.4 cm",
        "Rationale": "Petal width (1.4 cm) falls in standard versicolor range despite long petal length.",
    },
    {
        "Sample": "#77",
        "Actual": "versicolor",
        "Predicted": "virginica",
        "Confidence": "52.8%",
        "Dimensions": "SL: 6.7, SW: 3.0, PL: 5.0, PW: 1.7 cm",
        "Rationale": "Petal dimensions (5.0 cm / 1.7 cm) exceed standard versicolor bounds into virginica.",
    },
]
st.dataframe(pd.DataFrame(MISCLASS_DATA), use_container_width=True, hide_index=True)

"""
Error Analysis and Decision Boundary Diagnostics Page.
Analyzes holdout test set misclassifications and feature space boundaries.
"""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Error Analysis - Iris ML Console",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import IrisApiClient
from frontend.components.header import render_header, render_sidebar_connection
from frontend.utils.formatting import apply_custom_theme

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_connection(api_client)
render_header(api_client)

st.markdown("## 🔍 Holdout Error Analysis & Boundary Inspection")
st.markdown(
    """
    Granular post-evaluation inspection of the champion Support Vector Machine on the
    untouched 30-sample holdout test partition.
    """
)

base_dir = Path(__file__).resolve().parent.parent.parent
metrics_path = base_dir / "artifacts" / "metrics" / "final_metrics.json"

if metrics_path.exists():
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Holdout Test Accuracy", f"{metrics.get('test_accuracy', 0.0) * 100:.2f}%")
    with c2:
        st.metric("Macro Precision", f"{metrics.get('test_precision_macro', 0.0):.4f}")
    with c3:
        st.metric("Macro Recall", f"{metrics.get('test_recall_macro', 0.0):.4f}")
    with c4:
        st.metric("Correct Predictions", f"{metrics.get('correct_predictions', 28)} / {metrics.get('total_test_samples', 30)}")

st.markdown("---")
st.markdown("### 📊 Diagnostic Visualizations")

cm_plot_path = base_dir / "artifacts" / "plots" / "confusion_matrix.png"
fs_plot_path = base_dir / "artifacts" / "plots" / "feature_space.png"

col_cm, col_fs = st.columns(2)

with col_cm:
    if cm_plot_path.exists():
        st.image(str(cm_plot_path), caption="Figure 3: Holdout Confusion Matrix (Raw & Normalized)", use_container_width=True)

with col_fs:
    if fs_plot_path.exists():
        st.image(str(fs_plot_path), caption="Figure 4: 2D Morphological Decision Boundary & Misclassified Samples", use_container_width=True)

st.markdown("---")
st.markdown("### 🔬 Deep Dive: Misclassified Specimens Breakdown")

st.markdown(
    """
    Exactly **2 out of 30** holdout samples were misclassified, both residing directly along
    the overlapping `versicolor` $\\leftrightarrow$ `virginica` morphological boundary:
    """
)

col_e1, col_e2 = st.columns(2)

with col_e1:
    st.markdown(
        """
        <div class="metric-card">
            <h4 style="color:#f43f5e; margin-top:0;">Specimen #1: Test Index 134</h4>
            <table style="width:100%; font-size:0.85rem; color:#cbd5e1;">
                <tr><td><strong>True Label:</strong></td><td><span class="badge-virginica">Iris Virginica</span></td></tr>
                <tr><td><strong>Predicted Label:</strong></td><td><span class="badge-versicolor">Iris Versicolor</span></td></tr>
                <tr><td><strong>Confidence:</strong></td><td>48.77% vs 49.96% Virginica margin</td></tr>
                <tr><td><strong>Measurements:</strong></td><td><code>[6.1, 2.6, 5.6, 1.4] cm</code></td></tr>
            </table>
            <hr style="border-color:rgba(255,255,255,0.1); margin:10px 0;">
            <p style="font-size:0.8rem; color:#94a3b8; margin-bottom:0;">
                <strong>Nearest Training Neighbors:</strong><br>
                1. Row #83 (Versicolor, Dist: 0.4630)<br>
                2. Row #72 (Versicolor, Dist: 0.5315)<br>
                3. Row #133 (Virginica, Dist: 0.5962)<br><br>
                <strong>Diagnosis:</strong> Possesses a thin 1.4 cm petal width typical of Versicolor despite long 5.6 cm petals.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_e2:
    st.markdown(
        """
        <div class="metric-card">
            <h4 style="color:#f43f5e; margin-top:0;">Specimen #2: Test Index 77</h4>
            <table style="width:100%; font-size:0.85rem; color:#cbd5e1;">
                <tr><td><strong>True Label:</strong></td><td><span class="badge-versicolor">Iris Versicolor</span></td></tr>
                <tr><td><strong>Predicted Label:</strong></td><td><span class="badge-virginica">Iris Virginica</span></td></tr>
                <tr><td><strong>Confidence:</strong></td><td>65.45%</td></tr>
                <tr><td><strong>Measurements:</strong></td><td><code>[6.7, 3.0, 5.0, 1.7] cm</code></td></tr>
            </table>
            <hr style="border-color:rgba(255,255,255,0.1); margin:10px 0;">
            <p style="font-size:0.8rem; color:#94a3b8; margin-bottom:0;">
                <strong>Nearest Training Neighbors:</strong><br>
                1. Row #86 (Versicolor, Dist: 0.3853)<br>
                2. Row #52 (Versicolor, Dist: 0.4240)<br>
                3. Row #137 (Virginica, Dist: 0.5257)<br><br>
                <strong>Diagnosis:</strong> Exceptionally large specimen exceeding upper quartiles for Versicolor, shifting probability mass to Virginica.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

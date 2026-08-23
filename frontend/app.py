"""
IRIS ML - Production ML Inference Platform
Overview & Operations Hub
"""

import streamlit as st

st.set_page_config(
    page_title="IRIS ML - Production ML",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.api_client import IrisApiClient
from frontend.components.metrics_cards import render_kpi_card
from frontend.components.navigation import render_sidebar_navigation
from frontend.utils.formatting import apply_custom_theme

# Apply design system
apply_custom_theme()

# Unified Sidebar Navigation
api_client = IrisApiClient()
health_data = render_sidebar_navigation(api_client)

# Header
st.markdown(
    """
    <div style="padding-bottom: 8px;">
        <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">IRIS ML</div>
        <div style="font-size: 0.95rem; font-weight: 600; color: #94a3b8; margin-top: 1px;">Production ML Inference Platform</div>
        <div style="font-size: 0.82rem; color: #64748b; margin-top: 2px;">Real-time tabular classification, model monitoring, and evaluation.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Live Status Bar
is_online = api_client.check_connection()
model_ver = health_data.get("model_version", "1.0.0") if is_online else "1.0.0"

status_dot_html = '<span class="status-dot dot-green"></span> Production API Online' if is_online else '<span class="status-dot dot-red"></span> API Offline'

st.markdown(
    f"""
    <div class="status-bar">
        <div class="status-bar-item" style="color: {'#10b981' if is_online else '#ef4444'}; font-weight: 600;">
            {status_dot_html}
        </div>
        <span style="color: #334155;">|</span>
        <div class="status-bar-item">Model: <span style="color: #f8fafc; margin-left: 4px;">SVM v{model_ver}</span></div>
        <span style="color: #334155;">|</span>
        <div class="status-bar-item">Dataset: <span style="color: #f8fafc; margin-left: 4px;">Iris</span></div>
        <span style="color: #334155;">|</span>
        <div class="status-bar-item">Classes: <span style="color: #f8fafc; margin-left: 4px;">3</span></div>
        <span style="color: #334155;">|</span>
        <div class="status-bar-item">Features: <span style="color: #f8fafc; margin-left: 4px;">4</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Run a Prediction Section
st.markdown("### Run a Prediction")
col_pred_info, col_pred_btn = st.columns([3, 1])
with col_pred_info:
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
            Enter four biological flower measurements (sepal & petal length/width in cm) to classify a botanical specimen using the production Linear SVM model.
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_pred_btn:
    st.page_link("pages/1_Predict.py", label="Predict a specimen →")

st.markdown("<hr style='margin: 16px 0;'/>", unsafe_allow_html=True)

# Model Performance Section
st.markdown("### Model Performance")
k1, k2, k3 = st.columns(3)
with k1:
    render_kpi_card("CV Accuracy", "97.50% ± 2.04%", "5-Fold Stratified CV (N=120)")
with k2:
    render_kpi_card("Holdout Accuracy", "93.33%", "28/30 untouched test samples")
with k3:
    render_kpi_card("Inference", "<3 ms", "Server execution SLA")

st.markdown("<hr style='margin: 16px 0;'/>", unsafe_allow_html=True)

# System Capabilities Section (2x3 Grid)
st.markdown("### System Capabilities")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="cap-box">
            <div class="cap-title">Real-time inference</div>
            <div class="cap-desc">Sub-3ms REST inference pipeline with calibrated Platt scaling probabilities.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="cap-box">
            <div class="cap-title">Model monitoring</div>
            <div class="cap-desc">Continuous Prometheus telemetry, latency percentiles, and Welford running moments.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="cap-box">
            <div class="cap-title">Feature drift</div>
            <div class="cap-desc">Statistical distribution comparison using Two-Sample KS, Wasserstein, and PSI.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
with c4:
    st.markdown(
        """
        <div class="cap-box">
            <div class="cap-title">Model comparison</div>
            <div class="cap-desc">Automated 7-model tournament benchmark with stratified 5-fold cross-validation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        """
        <div class="cap-box">
            <div class="cap-title">Error analysis</div>
            <div class="cap-desc">Holdout confusion matrix, classification metrics, and boundary specimen analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c6:
    st.markdown(
        """
        <div class="cap-box">
            <div class="cap-title">Explainability</div>
            <div class="cap-desc">Model Card governance specifications and linear SVM feature weight attribution.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

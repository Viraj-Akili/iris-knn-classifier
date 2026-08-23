"""
IRIS ML - Model Governance & Specification Module
Model architecture overview, feature influence weights, and operational boundaries.
"""

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Model - IRIS ML",
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
render_header(api_client, title="Model", subtitle="Production model architecture, specifications, and feature influence.")

# Model Overview & Performance Row
col_ov, col_perf = st.columns(2)

with col_ov:
    st.markdown("### Model Overview")
    st.markdown(
        """
        <div class="cap-box">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">SVM v1.0.0</div>
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.8; font-family: 'JetBrains Mono', monospace;">
                • Status: <span style="color: #10b981; font-weight: 600;">Loaded (Active)</span><br>
                • Architecture: StandardScaler ➔ Linear SVC<br>
                • Training Strategy: 5-Fold Stratified CV (N=120)<br>
                • Dataset: Fisher's Iris Benchmark (150 samples)<br>
                • Target Classes: setosa, versicolor, virginica<br>
                • Features: sepal length, sepal width, petal length, petal width
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_perf:
    st.markdown("### Model Performance")
    p1, p2 = st.columns(2)
    with p1:
        render_kpi_card("CV Accuracy", "97.50% ± 2.04%", "5-Fold Stratified")
    with p2:
        render_kpi_card("Holdout Accuracy", "93.33%", "28/30 test samples")

    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 8px; line-height: 1.4;">
            Champion pipeline selected across 7 candidate algorithms. Fitted with calibrated Platt scaling for calibrated posterior probability estimation.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin: 14px 0;'/>", unsafe_allow_html=True)

# Feature Influence Section
st.markdown("### Feature Influence")
st.caption("Normalized linear decision boundary weight magnitudes across One-vs-Rest hyperplanes.")

WEIGHTS_DATA = [
    {"Feature": "Petal Length (cm)", "Importance Score": 0.468, "Impact Level": "Primary Separator"},
    {"Feature": "Petal Width (cm)", "Importance Score": 0.382, "Impact Level": "Primary Separator"},
    {"Feature": "Sepal Width (cm)", "Importance Score": 0.096, "Impact Level": "Secondary Stabilizer"},
    {"Feature": "Sepal Length (cm)", "Importance Score": 0.054, "Impact Level": "Minor Discriminator"},
]
df_weights = pd.DataFrame(WEIGHTS_DATA)

chart = (
    alt.Chart(df_weights)
    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=28)
    .encode(
        x=alt.X("Importance Score:Q", title="Decision Boundary Weight", axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b")),
        y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=11, labelFontWeight="bold", labelColor="#94a3b8")),
        color=alt.Color(
            "Importance Score:Q",
            scale=alt.Scale(scheme="tealblues"),
            legend=None,
        ),
        tooltip=["Feature", "Importance Score", "Impact Level"],
    )
    .properties(height=180)
)

st.altair_chart(chart, use_container_width=True)

st.markdown("<hr style='margin: 14px 0;'/>", unsafe_allow_html=True)

# Limitations & API Status Row
col_lim, col_api = st.columns(2)

with col_lim:
    st.markdown("### Limitations")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.6;">
            • <strong>Dataset Scope</strong>: Iris is a small tabular benchmark ($N=150$); not intended for real-world clinical or biological diagnosis.<br>
            • <strong>Overlapping Margin</strong>: Moderate boundary overlap exists between <em>versicolor</em> and <em>virginica</em> near $PL \approx 5.0\text{ cm}, PW \approx 1.5\text{ cm}$.<br>
            • <strong>Non-Causal Attribution</strong>: Feature coefficients reflect predictive association rather than causal biological importance.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_api:
    st.markdown("### API Status")
    is_online = api_client.check_connection()
    if is_online:
        st.markdown(
            """
            <div class="cap-box">
                <div style="display: flex; align-items: center; font-size: 0.85rem; color: #10b981; font-weight: 600; margin-bottom: 4px;">
                    <span class="status-dot dot-green"></span> Production API Online
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.5;">
                    • Endpoint: <code style="color: #60a5fa;">https://iris-ml-backend.onrender.com</code><br>
                    • Readiness: <span style="color: #10b981;">Ready (HTTP 200)</span><br>
                    • Model: <span style="color: #f8fafc;">SVM v1.0.0</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="cap-box">
                <div style="display: flex; align-items: center; font-size: 0.85rem; color: #ef4444; font-weight: 600; margin-bottom: 4px;">
                    <span class="status-dot dot-red"></span> API Offline
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8;">
                    Inference backend unreachable. Start the FastAPI server to restore connectivity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

"""
Iris ML Intelligence Dashboard - Main Entrypoint & Overview.
Streamlit frontend connecting to the FastAPI production ML inference backend.
"""

import streamlit as st

st.set_page_config(
    page_title="Iris ML Intelligence Console",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.api_client import IrisApiClient
from frontend.components.header import render_header, render_sidebar_connection
from frontend.components.metrics_cards import render_kpi_row, render_latency_kpi_row
from frontend.utils.formatting import apply_custom_theme

# Apply Custom Styling
apply_custom_theme()

# Initialize API Client
api_client = IrisApiClient()

# Render Sidebar & Top Header
render_sidebar_connection(api_client)
status_info = render_header(api_client)

st.markdown("## 🧭 Dashboard Navigation & Operations Hub")
st.markdown(
    """
    Welcome to the **Iris ML Intelligence & Operations Console**. This application connects directly
    to the production **FastAPI inference backend** to provide real-time model predictions,
    live telemetry, statistical data drift inspection, and experimental diagnostics.
    """
)

# Operational Overview Row
if status_info["is_connected"]:
    try:
        summary = api_client.get_observability_summary()
        st.markdown("### ⚡ Live System Telemetry")
        render_kpi_row(summary)

        st.markdown("#### ⏱️ Latency Percentiles (Sliding Buffer N=1000)")
        render_latency_kpi_row(summary.get("latency_statistics_ms", {}))
    except Exception as e:
        st.warning(f"Could not load live observability summary: {e}")
else:
    st.error(
        "⚠️ **FastAPI Backend is currently unreachable.**\n\n"
        "Please ensure the backend server is running locally:\n\n"
        "```bash\n"
        "uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload\n"
        "```"
    )

st.markdown("---")
st.markdown("### 📑 Operational Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="margin-top:0; font-size:1.2rem; color:#818cf8;">🔮 1. Real-Time Inference</h3>
            <p style="font-size:0.85rem; color:#94a3b8;">
                Interactive prediction interface with morphological presets, calibrated class probabilities, and sub-millisecond latency tracking.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="margin-top:0; font-size:1.2rem; color:#38bdf8;">📊 2. Live Observability</h3>
            <p style="font-size:0.85rem; color:#94a3b8;">
                Prometheus metrics, real-time prediction distribution, confidence tiering, and privacy-safe online feature aggregation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="margin-top:0; font-size:1.2rem; color:#f43f5e;">🧪 3. Data Drift Analysis</h3>
            <p style="font-size:0.85rem; color:#94a3b8;">
                Two-Sample Kolmogorov-Smirnov test, Wasserstein physical distance, and Population Stability Index (PSI) drift monitoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="margin-top:0; font-size:1.2rem; color:#eab308;">🏆 4. Model Tournament</h3>
            <p style="font-size:0.85rem; color:#94a3b8;">
                7-algorithm 5-fold cross-validation tournament leaderboard, variance analysis, and champion SVM selection rationale.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="margin-top:0; font-size:1.2rem; color:#ec4899;">🔍 5. Error Analysis</h3>
            <p style="font-size:0.85rem; color:#94a3b8;">
                Confusion matrices, 2D decision boundary visualization, and granular nearest-neighbor diagnostic breakdown of misclassified samples.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="margin-top:0; font-size:1.2rem; color:#10b981;">📋 6. Model Card & XAI</h3>
            <p style="font-size:0.85rem; color:#94a3b8;">
                Formal ML model governance specification, feature importance weights, dataset constraints, and operational limitations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.info(
    "💡 **Architecture Guarantee**: This frontend communicates strictly over HTTP with the FastAPI REST API. "
    "No Scikit-Learn models are duplicated or loaded directly in Streamlit."
)

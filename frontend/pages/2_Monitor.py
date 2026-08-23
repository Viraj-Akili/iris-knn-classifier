"""
IRIS ML - Monitor Module
Runtime performance, operational telemetry, and prediction distributions.
"""

import streamlit as st

st.set_page_config(
    page_title="Monitor - IRIS ML",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import IrisApiClient
from frontend.components.charts import (
    render_class_distribution_chart,
    render_confidence_chart,
)
from frontend.components.header import render_header
from frontend.components.metrics_cards import render_kpi_card
from frontend.components.navigation import render_sidebar_navigation
from frontend.utils.formatting import apply_custom_theme

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_navigation(api_client)
render_header(api_client, title="Production monitoring", subtitle="Runtime performance and prediction telemetry.")

telemetry = {}
try:
    if api_client.check_connection():
        telemetry = api_client.get_observability_summary()
except Exception:
    pass

if telemetry:
    total_reqs = telemetry.get("total_requests", 0)
    failed_reqs = telemetry.get("failed_requests", 0)
    success_rate = 100.0 if total_reqs == 0 else max(0.0, (total_reqs - failed_reqs) / total_reqs * 100.0)

    lat_stats = telemetry.get("latency_percentiles", {})
    p50 = lat_stats.get("p50_ms", 0.0)
    p95 = lat_stats.get("p95_ms", 0.0)
    p99 = lat_stats.get("p99_ms", 0.0)

    # Top KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Requests", f"{total_reqs:,}", "Total served")
    with col2:
        render_kpi_card("Success Rate", f"{success_rate:.1f}%", f"{failed_reqs} errors")
    with col3:
        render_kpi_card("P50 Latency", f"{p50:.2f} ms", "Median")
    with col4:
        render_kpi_card("P95 Latency", f"{p95:.2f} ms", "Tail target")
    with col5:
        render_kpi_card("P99 Latency", f"{p99:.2f} ms", "Peak tail")

    st.markdown("<hr style='margin: 14px 0;'/>", unsafe_allow_html=True)

    # 2-Column Clean Chart Layout
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Predictions")
        st.caption("Distribution of predicted classes across runtime inferences.")
        class_counts = telemetry.get("prediction_distribution", {})
        render_class_distribution_chart(class_counts)

    with c2:
        st.markdown("### Confidence")
        st.caption("Distribution of prediction confidence across calibration tiers.")
        conf_dist = telemetry.get("confidence_distribution", {})
        render_confidence_chart(conf_dist)

    st.markdown("<hr style='margin: 14px 0;'/>", unsafe_allow_html=True)

    # System Health Section
    st.markdown("### System Health")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(
            """
            <div class="cap-box">
                <div class="cap-title"><span class="status-dot dot-green"></span> API</div>
                <div class="cap-desc">FastAPI inference service operational.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            """
            <div class="cap-box">
                <div class="cap-title"><span class="status-dot dot-green"></span> Model</div>
                <div class="cap-desc">Support Vector Machine v1.0.0 loaded in memory.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with h3:
        st.markdown(
            """
            <div class="cap-box">
                <div class="cap-title"><span class="status-dot dot-green"></span> Telemetry</div>
                <div class="cap-desc">Prometheus metrics and Welford aggregation active.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div style="background-color: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 18px; color: #94a3b8; font-size: 0.85rem;">
            Unable to reach the inference API. Start the backend service to stream live operational metrics.
        </div>
        """,
        unsafe_allow_html=True,
    )

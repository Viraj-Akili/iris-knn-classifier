"""
Monitoring Module for IRIS ML PLATFORM.
Real-time operational telemetry, Prometheus metrics, and latency percentiles.
"""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Monitoring - IRIS ML PLATFORM",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import IrisApiClient
from frontend.components.charts import (
    render_class_distribution_chart,
    render_confidence_chart,
    render_feature_distribution_chart,
)
from frontend.components.header import render_header, render_sidebar_connection
from frontend.components.metrics_cards import render_kpi_card
from frontend.utils.formatting import apply_custom_theme

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_connection(api_client)
render_header(api_client)

st.markdown("## Production monitoring")
st.caption("Live operational telemetry and statistical distributions from the FastAPI backend.")

try:
    telemetry = api_client.get_observability_summary()
except Exception as e:
    st.error(f"Could not load telemetry: {e}")
    telemetry = {}

if telemetry:
    total_reqs = telemetry.get("total_requests", 0)
    failed_reqs = telemetry.get("failed_requests", 0)
    success_rate = 100.0 if total_reqs == 0 else max(0.0, (total_reqs - failed_reqs) / total_reqs * 100.0)

    lat_stats = telemetry.get("latency_percentiles", {})
    p50 = lat_stats.get("p50_ms", 0.0)
    p90 = lat_stats.get("p90_ms", 0.0)
    p95 = lat_stats.get("p95_ms", 0.0)
    p99 = lat_stats.get("p99_ms", 0.0)
    min_lat = lat_stats.get("min_ms", 0.0)
    max_lat = lat_stats.get("max_ms", 0.0)

    # Compact KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Requests", f"{total_reqs:,}", "Total served")
    with col2:
        render_kpi_card("Success Rate", f"{success_rate:.1f}%", f"{failed_reqs} errors")
    with col3:
        render_kpi_card("Mean Latency", f"{p50:.2f} ms", f"Min: {min_lat:.1f}ms")
    with col4:
        render_kpi_card("p95 Latency", f"{p95:.2f} ms", "Tail target")
    with col5:
        render_kpi_card("p99 Latency", f"{p99:.2f} ms", f"Max: {max_lat:.1f}ms")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts: 2-column layout
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Prediction distribution")
        class_counts = telemetry.get("prediction_distribution", {})
        render_class_distribution_chart(class_counts)

    with c2:
        st.markdown("### Confidence distribution")
        conf_dist = telemetry.get("confidence_distribution", {})
        render_confidence_chart(conf_dist)

    st.markdown("---")

    # Privacy-Safe Feature Running Stats
    st.markdown("### Privacy-safe feature monitoring")
    st.caption("Online running moments computed via Welford's algorithm across incoming prediction requests.")

    feature_stats = telemetry.get("feature_statistics", {})
    if feature_stats:
        render_feature_distribution_chart(feature_stats)

        rows = []
        for feat_name, stats in feature_stats.items():
            if stats.get("count", 0) > 0:
                rows.append({
                    "Feature": feat_name.replace("_", " ").title() + " (cm)",
                    "Samples": stats.get("count", 0),
                    "Mean (cm)": f"{stats.get('mean', 0.0):.3f}",
                    "Std Dev (cm)": f"{stats.get('std', 0.0):.3f}",
                    "Min (cm)": f"{stats.get('min', 0.0):.2f}",
                    "Max (cm)": f"{stats.get('max', 0.0):.2f}",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No incoming feature statistics recorded yet.")

    # Raw Prometheus Exposition
    with st.expander("Prometheus metrics exposition (/metrics)"):
        try:
            prom_text = api_client.get_prometheus_metrics()
            st.code(prom_text, language="text")
        except Exception as e:
            st.error(f"Failed to load Prometheus endpoint: {e}")
else:
    st.warning("Telemetry data currently unavailable from backend.")

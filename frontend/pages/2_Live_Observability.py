"""
Live ML Observability and Real-Time Telemetry Dashboard.
Consumes /observability/summary and Prometheus metrics from the FastAPI backend.
"""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Live Observability - Iris ML Console",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import ApiConnectionError, IrisApiClient
from frontend.components.charts import (
    render_class_distribution_chart,
    render_confidence_chart,
    render_feature_distribution_chart,
)
from frontend.components.header import render_header, render_sidebar_connection
from frontend.components.metrics_cards import render_kpi_row, render_latency_kpi_row
from frontend.utils.formatting import apply_custom_theme

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_connection(api_client)
render_header(api_client)

st.markdown("## 📊 Live ML Observability & Runtime Telemetry")
st.markdown(
    """
    Monitor real-time inference throughput, exact sliding-window latency percentiles,
    prediction class frequencies, confidence tiering, and privacy-safe input feature distributions.
    """
)

try:
    summary = api_client.get_observability_summary()

    st.markdown("### ⚡ Live System KPIs")
    render_kpi_row(summary)

    st.markdown("### ⏱️ Latency Percentiles (Sliding Buffer N=1000)")
    render_latency_kpi_row(summary.get("latency_statistics_ms", {}))
    st.caption("Empirical percentiles computed via bounded in-memory sliding buffer. Latency includes model pipeline inference and feature mapping.")

    st.markdown("---")
    col_dist, col_conf = st.columns(2)

    with col_dist:
        st.markdown("### 🌸 Prediction Class Distribution")
        render_class_distribution_chart(summary.get("predictions_by_class", {}))
        st.caption("Distribution of predicted flower species across active session inferences.")

    with col_conf:
        st.markdown("### 🎯 Confidence Tier Breakdown")
        render_confidence_chart(summary.get("confidence_distribution", {}))
        st.caption("Grouped by: High (>= 90%), Medium (70-90%), Low (< 70%).")

    st.markdown("---")
    st.markdown("### 🌿 Privacy-Safe Online Input Feature Aggregation")
    st.markdown(
        """
        Calculated online using **Welford's Algorithm**. Raw user inputs are not stored in memory
        or exposed through Prometheus metric labels.
        """
    )

    feat_aggs = summary.get("feature_aggregates", {})
    if feat_aggs:
        records = []
        for feat_name, stat in feat_aggs.items():
            records.append({
                "Feature Name": feat_name.replace("_", " ").title() + " (cm)",
                "Sample Count (N)": stat.get("count", 0),
                "Running Mean (cm)": f"{stat.get('mean', 0.0):.3f}",
                "Running Std Dev (cm)": f"{stat.get('std', 0.0):.3f}",
                "Observed Min (cm)": f"{stat.get('min', 0.0):.2f}",
                "Observed Max (cm)": f"{stat.get('max', 0.0):.2f}",
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        render_feature_distribution_chart(feat_aggs)
    else:
        st.info("No incoming feature statistics recorded yet.")

    st.markdown("---")
    with st.expander("📡 Raw Prometheus Metric Exposition (GET /metrics)", expanded=False):
        try:
            prom_text = api_client.get_prometheus_metrics()
            st.code(prom_text, language="text")
        except Exception as e:
            st.warning(f"Could not retrieve raw Prometheus metrics: {e}")

    st.info(
        "⚠️ **ML Confidence Limitation**: Model confidence score reflects calibrated softmax / Platt output probabilities "
        "and does not strictly guarantee ground-truth correctness, especially under out-of-distribution inputs."
    )

except ApiConnectionError as e:
    st.error(f"🔴 Backend Connection Error: {e}")
except Exception as e:
    st.error(f"🚨 Error loading observability data: {e}")

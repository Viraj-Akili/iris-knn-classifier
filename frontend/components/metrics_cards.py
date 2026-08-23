"""
Minimalist KPI and telemetry metric components for IRIS ML PLATFORM.
"""

from typing import Any

import streamlit as st


def render_kpi_card(label: str, value: str, subtext: str | None = None) -> None:
    """Render a single minimalist KPI card."""
    sub_html = f'<div class="kpi-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(
    model_name: str,
    cv_accuracy: float,
    holdout_accuracy: float,
    p95_latency_ms: float,
) -> None:
    """Render top-level KPI row for Overview and Dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Model", model_name, "Linear SVM Champion")
    with col2:
        render_kpi_card("CV Accuracy", f"{cv_accuracy * 100.0:.2f}%", "5-Fold Stratified")
    with col3:
        render_kpi_card("Holdout Accuracy", f"{holdout_accuracy * 100.0:.2f}%", "Untouched Test Set")
    with col4:
        render_kpi_card("p95 Latency", f"{p95_latency_ms:.2f} ms", "Latency SLA Target")


def render_latency_kpi_row(telemetry: dict[str, Any]) -> None:
    """Render compact KPI row for operational latencies."""
    lat_stats = telemetry.get("latency_percentiles", {})
    p50 = lat_stats.get("p50_ms", 0.0)
    p95 = lat_stats.get("p95_ms", 0.0)
    p99 = lat_stats.get("p99_ms", 0.0)
    total_reqs = telemetry.get("total_requests", 0)
    failed_reqs = telemetry.get("failed_requests", 0)

    success_rate = 100.0 if total_reqs == 0 else max(0.0, (total_reqs - failed_reqs) / total_reqs * 100.0)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Requests", f"{total_reqs:,}", "Total served")
    with col2:
        render_kpi_card("Success Rate", f"{success_rate:.1f}%", f"{failed_reqs} errors")
    with col3:
        render_kpi_card("p50 Latency", f"{p50:.2f} ms", "Median")
    with col4:
        render_kpi_card("p95 Latency", f"{p95:.2f} ms", "Tail target")
    with col5:
        render_kpi_card("p99 Latency", f"{p99:.2f} ms", "Peak tail")

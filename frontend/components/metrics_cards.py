"""
KPI metric card renderers for system throughput, latency percentiles, and error counts.
"""

from typing import Any

import streamlit as st


def render_kpi_row(summary: dict[str, Any]) -> None:
    """Render top-level operational KPIs (Requests, Success, Failures, Mean Latency)."""
    c1, c2, c3, c4 = st.columns(4)

    total_req = summary.get("total_requests", 0)
    succ_req = summary.get("successful_predictions", 0)
    fail_req = summary.get("failed_requests", 0)
    uptime_sec = summary.get("uptime_seconds", 0.0)

    lat_stats = summary.get("latency_statistics_ms", {})
    mean_lat = lat_stats.get("mean_ms", 0.0)

    success_rate = (succ_req / total_req * 100.0) if total_req > 0 else 100.0

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Predictions</div>
                <div class="metric-value">{total_req:,}</div>
                <div class="metric-subtext">{succ_req} Successful ({success_rate:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mean Inference Latency</div>
                <div class="metric-value">{mean_lat:.2f} <span style="font-size:1rem; font-weight:500;">ms</span></div>
                <div class="metric-subtext">p50: {lat_stats.get('p50_ms', 0.0):.2f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Failed Requests</div>
                <div class="metric-value" style="color: {'#ef4444' if fail_req > 0 else '#10b981'};">{fail_req}</div>
                <div class="metric-subtext">Validation & Error Count</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Service Uptime</div>
                <div class="metric-value">{uptime_sec:.1f} <span style="font-size:1rem; font-weight:500;">s</span></div>
                <div class="metric-subtext">In-Memory Singleton</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_latency_kpi_row(lat_stats: dict[str, float]) -> None:
    """Render granular latency percentiles (p50, p90, p95, p99, Min, Max)."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric("p50 (Median)", f"{lat_stats.get('p50_ms', 0.0):.2f} ms")
    with c2:
        st.metric("p90", f"{lat_stats.get('p90_ms', lat_stats.get('p95_ms', 0.0)):.2f} ms")
    with c3:
        st.metric("p95", f"{lat_stats.get('p95_ms', 0.0):.2f} ms")
    with c4:
        st.metric("p99", f"{lat_stats.get('p99_ms', 0.0):.2f} ms")
    with c5:
        st.metric("Min", f"{lat_stats.get('min_ms', 0.0):.2f} ms")
    with c6:
        st.metric("Max", f"{lat_stats.get('max_ms', 0.0):.2f} ms")

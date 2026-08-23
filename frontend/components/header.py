"""
System Status Header and Sidebar Connection Controls.
"""

from datetime import UTC, datetime
from typing import Any

import streamlit as st

from frontend.api_client import IrisApiClient
from frontend.utils.formatting import render_status_badge


def render_header(client: IrisApiClient) -> dict[str, Any]:
    """
    Render live system status banner at top of dashboard pages.
    Queries FastAPI backend dynamically and returns system state.
    """
    is_connected = False
    is_ready = False
    model_name = "Unavailable"
    model_version = "N/A"
    uptime_str = "0s"
    checks_dict = {}

    try:
        client.get_health()
        is_connected = True
        readiness_data = client.get_readiness()
        is_ready = readiness_data.get("model_loaded", False)
        checks_dict = readiness_data.get("checks", {})

        if is_ready:
            model_info = client.get_model_info()
            model_name = model_info.get("model_name", "Support Vector Machine")
            model_version = model_info.get("model_version", "1.0.0")

        # Get uptime if available
        summary = client.get_observability_summary()
        uptime_sec = summary.get("uptime_seconds", 0.0)
        mins, secs = divmod(int(uptime_sec), 60)
        hours, mins = divmod(mins, 60)
        uptime_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}s"

    except Exception:
        is_connected = False
        is_ready = False

    current_time = datetime.now(UTC).strftime("%H:%M:%S UTC")

    # Render Header Banner
    col_title, col_status = st.columns([3, 2])
    with col_title:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
                <h1 style="margin:0; font-size:1.9rem; font-weight:800; color:#f8fafc; letter-spacing:-0.02em;">
                    Iris ML Intelligence & Operations Console
                </h1>
            </div>
            <div style="color:#94a3b8; font-size:0.88rem; margin-bottom:12px;">
                Production Machine Learning Inference, Observability, and Statistical Drift Detection
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_status:
        badge_html = render_status_badge(is_ready=is_ready, is_alive=is_connected)
        st.markdown(
            f"""
            <div style="text-align:right; margin-bottom:8px;">
                {badge_html}
            </div>
            <div style="text-align:right; color:#64748b; font-size:0.78rem; font-family:'JetBrains Mono', monospace;">
                Model: <strong style="color:#cbd5e1;">{model_name}</strong> | v{model_version} | Refreshed: {current_time}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    return {
        "is_connected": is_connected,
        "is_ready": is_ready,
        "model_name": model_name,
        "model_version": model_version,
        "uptime": uptime_str,
        "checks": checks_dict,
        "timestamp": current_time,
    }


def render_sidebar_connection(client: IrisApiClient) -> None:
    """Render sidebar connection status and refresh configuration."""
    with st.sidebar:
        st.markdown("### 🔌 API Backend Connection")
        st.markdown(f"**Endpoint**: `{client.base_url}`")

        is_alive = client.check_connection()
        if is_alive:
            st.success("🟢 FastAPI Backend Online")
        else:
            st.error("🔴 FastAPI Backend Unreachable")
            st.caption("Run: `uvicorn app.main:app --host 0.0.0.0 --port 8000`")

        st.markdown("---")
        st.markdown("### ⚙️ Dashboard Controls")

        st.selectbox(
            "Auto-Refresh Rate",
            options=["Manual Only", "5 seconds", "10 seconds", "30 seconds"],
            index=0,
        )

        if st.button("🔄 Manual Refresh Now", use_container_width=True):
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#64748b; text-align:center;">
                <strong>Enterprise ML Operations v1.0.0</strong><br>
                FastAPI • Scikit-Learn • Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )

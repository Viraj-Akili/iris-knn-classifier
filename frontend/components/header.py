"""
Header and sidebar components for IRIS ML PLATFORM.
"""

from typing import Any

import streamlit as st

from frontend.api_client import IrisApiClient


def render_sidebar_connection(client: IrisApiClient) -> dict[str, Any]:
    """Render minimal sidebar branding and system health monitor."""
    with st.sidebar:
        st.markdown("### IRIS ML PLATFORM")
        st.caption("ML inference & operations")
        st.markdown("---")

        # Query backend health
        is_online = client.check_connection()
        health_data = {}
        model_name = "SVM"
        model_ver = "1.0.0"

        if is_online:
            try:
                health_data = client.get_health()
                model_ver = health_data.get("model_version", "1.0.0")
            except Exception:
                pass

            st.markdown(
                f"""
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600; margin-bottom: 6px;">System</div>
                <div style="display: flex; align-items: center; font-size: 0.85rem; color: #10b981; font-weight: 600; margin-bottom: 4px;">
                    <span class="status-dot dot-green"></span> API Online
                </div>
                <div style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">
                    {model_name} v{model_ver}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600; margin-bottom: 6px;">System</div>
                <div style="display: flex; align-items: center; font-size: 0.85rem; color: #ef4444; font-weight: 600; margin-bottom: 4px;">
                    <span class="status-dot dot-red"></span> API Offline
                </div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                    FastAPI backend unreachable.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        return health_data


def render_header(client: IrisApiClient) -> None:
    """Render top-level page header with status indicator."""
    is_online = client.check_connection()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            """
            <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">
                IRIS ML PLATFORM
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 2px;">
                Real-time classification & model monitoring
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if is_online:
            st.markdown(
                """
                <div style="text-align: right; padding-top: 6px;">
                    <span class="meta-tag" style="border-color: #10b981; color: #10b981;">
                        <span class="status-dot dot-green"></span> Production API Online
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="text-align: right; padding-top: 6px;">
                    <span class="meta-tag" style="border-color: #ef4444; color: #ef4444;">
                        <span class="status-dot dot-red"></span> API Disconnected
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='margin: 16px 0 24px 0;'/>", unsafe_allow_html=True)

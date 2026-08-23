"""
Unified Sidebar Navigation Component for IRIS ML.
Provides structured, clean developer-tool navigation without default Streamlit noise.
"""

from typing import Any

import streamlit as st

from frontend.api_client import IrisApiClient


def render_sidebar_navigation(client: IrisApiClient) -> dict[str, Any]:
    """Render unified product identity, workspace links, and system status in sidebar."""
    with st.sidebar:
        # Product Brand Header
        st.markdown(
            """
            <div style="padding: 2px 0 10px 0;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">IRIS ML</div>
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 500;">Production ML</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Workspace Navigation
        st.markdown(
            '<div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; margin: 10px 0 4px 0;">Workspace</div>',
            unsafe_allow_html=True,
        )
        st.page_link("app.py", label="Overview")
        st.page_link("pages/1_Predict.py", label="Predict")
        st.page_link("pages/2_Monitor.py", label="Monitor")
        st.page_link("pages/3_Drift.py", label="Drift")
        st.page_link("pages/4_Models.py", label="Models")
        st.page_link("pages/5_Evaluation.py", label="Evaluation")

        st.markdown("<hr style='margin: 12px 0; border-color: #1e293b;'/>", unsafe_allow_html=True)

        # System Navigation
        st.markdown(
            '<div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; margin: 0 0 4px 0;">System</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/6_Model.py", label="Model")

        st.markdown("<hr style='margin: 12px 0; border-color: #1e293b;'/>", unsafe_allow_html=True)

        # System Status Footer
        is_online = client.check_connection()
        health_data = {}
        model_ver = "1.0.0"

        if is_online:
            try:
                health_data = client.get_health()
                model_ver = health_data.get("model_version", "1.0.0")
            except Exception:
                pass

            st.markdown(
                f"""
                <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; margin-bottom: 4px;">Status</div>
                <div style="display: flex; align-items: center; font-size: 0.8rem; color: #10b981; font-weight: 600; margin-bottom: 2px;">
                    <span class="status-dot dot-green"></span> API Online
                </div>
                <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #64748b;">
                    SVM v{model_ver}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; margin-bottom: 4px;">Status</div>
                <div style="display: flex; align-items: center; font-size: 0.8rem; color: #ef4444; font-weight: 600; margin-bottom: 2px;">
                    <span class="status-dot dot-red"></span> API Offline
                </div>
                <div style="font-size: 0.72rem; color: #64748b;">
                    Backend unreachable
                </div>
                """,
                unsafe_allow_html=True,
            )

        return health_data

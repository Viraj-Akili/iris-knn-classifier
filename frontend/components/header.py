"""
Page header component for IRIS ML.
"""

import streamlit as st

from frontend.api_client import IrisApiClient


def render_header(client: IrisApiClient, title: str | None = None, subtitle: str | None = None) -> None:
    """Render top-level page header with status indicator."""
    is_online = client.check_connection()

    if title:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"""
                <div style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">
                    {title}
                </div>
                """ + (f'<div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">{subtitle}</div>' if subtitle else ''),
                unsafe_allow_html=True,
            )
        with col2:
            if is_online:
                st.markdown(
                    """
                    <div style="text-align: right; padding-top: 4px;">
                        <span style="display: inline-flex; align-items: center; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; border-radius: 4px; background-color: #0f172a; border: 1px solid #10b981; color: #10b981;">
                            <span class="status-dot dot-green"></span> Online
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="text-align: right; padding-top: 4px;">
                        <span style="display: inline-flex; align-items: center; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; border-radius: 4px; background-color: #0f172a; border: 1px solid #ef4444; color: #ef4444;">
                            <span class="status-dot dot-red"></span> Offline
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<hr style='margin: 12px 0 16px 0;'/>", unsafe_allow_html=True)

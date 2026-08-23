"""
Design system, typography, and styling utilities for IRIS ML.
Modern, minimalist developer-tool aesthetic (Linear, Vercel, Stripe).
"""

import streamlit as st


def apply_custom_theme() -> None:
    """Inject clean, minimalist CSS design system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #f1f5f9;
        }

        code, pre, .mono {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Completely hide Streamlit's default unstyled sidebar navigation */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #080c14;
            border-right: 1px solid #1e293b;
            padding-top: 1.2rem;
        }

        [data-testid="stSidebar"] hr {
            border-color: #1e293b;
            margin: 12px 0;
        }

        /* Sidebar PageLink styling */
        [data-testid="stSidebar"] a {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            padding: 6px 10px !important;
            transition: all 0.15s ease !important;
            text-decoration: none !important;
        }

        [data-testid="stSidebar"] a:hover {
            background-color: #1e293b !important;
            color: #f8fafc !important;
        }

        /* Metric / KPI Card */
        .kpi-card {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 8px;
        }

        .kpi-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 2px;
        }

        .kpi-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #f8fafc;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: -0.02em;
        }

        .kpi-sub {
            font-size: 0.72rem;
            color: #64748b;
            margin-top: 2px;
        }

        /* Capability Box in Overview */
        .cap-box {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 14px;
            height: 100%;
        }

        .cap-title {
            font-size: 0.88rem;
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 4px;
        }

        .cap-desc {
            font-size: 0.78rem;
            color: #94a3b8;
            line-height: 1.4;
        }

        /* Status Bar Container */
        .status-bar {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 0.8rem;
            margin-bottom: 18px;
        }

        .status-bar-item {
            display: flex;
            align-items: center;
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
        }

        /* Prediction Result Panel */
        .prediction-panel {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px 24px;
            margin-top: 14px;
            margin-bottom: 18px;
        }

        .prediction-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            font-weight: 600;
        }

        .prediction-title {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            text-transform: uppercase;
            margin: 4px 0 10px 0;
        }

        .pred-setosa { color: #10b981; }
        .pred-versicolor { color: #3b82f6; }
        .pred-virginica { color: #a855f7; }

        /* Status Dots */
        .status-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .dot-green { background-color: #10b981; }
        .dot-amber { background-color: #f59e0b; }
        .dot-red { background-color: #ef4444; }

        /* Probability Bars */
        .prob-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.82rem;
        }

        .prob-bar-bg {
            flex-grow: 1;
            height: 8px;
            background-color: #1e293b;
            border-radius: 4px;
            margin: 0 12px;
            overflow: hidden;
        }

        .prob-bar-fill {
            height: 100%;
            border-radius: 4px;
        }

        /* Buttons */
        .stButton button {
            background-color: #2563eb !important;
            color: #ffffff !important;
            border: 1px solid #3b82f6 !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            font-size: 0.88rem !important;
            transition: all 0.15s ease !important;
        }

        .stButton button:hover {
            background-color: #1d4ed8 !important;
            border-color: #60a5fa !important;
        }

        /* Typography */
        h1, h2, h3 {
            letter-spacing: -0.02em;
            color: #f8fafc;
            font-weight: 700;
        }

        /* Dividers */
        hr {
            border: 0;
            border-top: 1px solid #1e293b;
            margin: 18px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_latency(ms: float) -> str:
    """Format milliseconds cleanly."""
    if ms < 1.0:
        return f"{ms * 1000.0:.0f} µs"
    return f"{ms:.2f} ms"


def format_percentage(val: float) -> str:
    """Format float or proportion as percentage."""
    if val <= 1.0:
        return f"{val * 100.0:.1f}%"
    return f"{val:.1f}%"


def render_status_badge(status: str) -> None:
    """Render semantic status chip."""
    status_lower = status.lower()
    if "online" in status_lower or "ready" in status_lower or "stable" in status_lower:
        st.markdown('<span class="status-dot dot-green"></span> <span>Healthy</span>', unsafe_allow_html=True)
    elif "warn" in status_lower:
        st.markdown('<span class="status-dot dot-amber"></span> <span>Warning</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot dot-red"></span> <span>Offline</span>', unsafe_allow_html=True)


def render_species_badge(species: str) -> None:
    """Render species label with color theme."""
    sp = species.lower()
    col = "#3b82f6"
    if "setosa" in sp:
        col = "#10b981"
    elif "virginica" in sp:
        col = "#a855f7"
    st.markdown(f'<span style="color: {col}; font-weight: 600;">{species.capitalize()}</span>', unsafe_allow_html=True)

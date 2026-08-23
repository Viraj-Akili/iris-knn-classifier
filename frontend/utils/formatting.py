"""
Design system, typography, and styling utilities for IRIS ML PLATFORM.
Minimalist, technical UI inspired by modern developer platforms (Linear, Vercel, Stripe).
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

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0b0f19;
            border-right: 1px solid #1e293b;
        }

        [data-testid="stSidebarNav"] {
            padding-top: 1rem;
        }

        /* Metric Container */
        .kpi-card {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }

        .kpi-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .kpi-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #f8fafc;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: -0.02em;
        }

        .kpi-sub {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 2px;
        }

        /* Prediction Result Panel */
        .prediction-panel {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 24px;
            margin-top: 16px;
            margin-bottom: 20px;
        }

        .prediction-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #94a3b8;
            font-weight: 600;
        }

        .prediction-title {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            text-transform: uppercase;
            margin: 6px 0 12px 0;
        }

        .pred-setosa { color: #10b981; }
        .pred-versicolor { color: #3b82f6; }
        .pred-virginica { color: #a855f7; }

        /* Status Dot */
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .dot-green { background-color: #10b981; }
        .dot-amber { background-color: #f59e0b; }
        .dot-red { background-color: #ef4444; }

        /* Meta Tag */
        .meta-tag {
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            background-color: #1e293b;
            color: #94a3b8;
            border: 1px solid #334155;
            margin-right: 6px;
        }

        /* Probability Bar */
        .prob-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.85rem;
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

        /* Button Styling */
        .stButton button {
            background-color: #2563eb !important;
            color: #ffffff !important;
            border: 1px solid #3b82f6 !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            padding: 8px 20px !important;
            transition: all 0.15s ease !important;
        }

        .stButton button:hover {
            background-color: #1d4ed8 !important;
            border-color: #60a5fa !important;
        }

        /* Clean Headers */
        h1, h2, h3 {
            letter-spacing: -0.02em;
            color: #f8fafc;
        }

        /* Divider */
        hr {
            border: 0;
            border-top: 1px solid #1e293b;
            margin: 20px 0;
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

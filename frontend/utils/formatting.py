"""
Visual styling, theme injection, and formatting utilities for ML Dashboard.
"""

import streamlit as st


def apply_custom_theme() -> None:
    """Inject polished, modern CSS styling for an enterprise ML operations console."""
    st.markdown(
        """
        <style>
        /* Modern ML Console Theme */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        code, pre, .mono-text {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Metric Card Container */
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.07) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .metric-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }

        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.02em;
        }

        .metric-subtext {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 4px;
        }

        /* Prediction Result Box */
        .prediction-hero {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border: 1px solid #4338ca;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(67, 56, 202, 0.2);
        }

        .prediction-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #a5b4fc;
            font-weight: 600;
        }

        .prediction-class {
            font-size: 2.8rem;
            font-weight: 800;
            text-transform: capitalize;
            color: #ffffff;
            margin: 8px 0;
            letter-spacing: -0.03em;
        }

        /* Badges */
        .status-badge-ready {
            display: inline-flex;
            align-items: center;
            background-color: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 9999px;
            padding: 4px 12px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }

        .status-badge-offline {
            display: inline-flex;
            align-items: center;
            background-color: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 9999px;
            padding: 4px 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10b981;
            margin-right: 6px;
            box-shadow: 0 0 8px #10b981;
        }

        .pulse-dot-red {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #ef4444;
            margin-right: 6px;
            box-shadow: 0 0 8px #ef4444;
        }

        /* Species Badges */
        .badge-setosa {
            background-color: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.4);
            border-radius: 6px;
            padding: 2px 8px;
            font-weight: 600;
        }
        .badge-versicolor {
            background-color: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.4);
            border-radius: 6px;
            padding: 2px 8px;
            font-weight: 600;
        }
        .badge-virginica {
            background-color: rgba(168, 85, 247, 0.15);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.4);
            border-radius: 6px;
            padding: 2px 8px;
            font-weight: 600;
        }

        /* Metadata Chip */
        .meta-chip {
            display: inline-block;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.8rem;
            color: #cbd5e1;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(is_ready: bool, is_alive: bool) -> str:
    """Generate HTML badge for system status."""
    if is_ready:
        return '<span class="status-badge-ready"><span class="pulse-dot"></span>SERVICE READY</span>'
    elif is_alive:
        return '<span class="status-badge-ready" style="color:#f59e0b; border-color:#f59e0b;"><span class="pulse-dot" style="background:#f59e0b;"></span>DEGRADED</span>'
    else:
        return '<span class="status-badge-offline"><span class="pulse-dot-red"></span>BACKEND OFFLINE</span>'


def render_species_badge(species: str) -> str:
    """Generate styled species badge HTML."""
    s_lower = species.lower()
    if s_lower == "setosa":
        return '<span class="badge-setosa">Iris Setosa</span>'
    elif s_lower == "versicolor":
        return '<span class="badge-versicolor">Iris Versicolor</span>'
    elif s_lower == "virginica":
        return '<span class="badge-virginica">Iris Virginica</span>'
    return f'<span class="meta-chip">{species}</span>'


def format_latency(ms: float) -> str:
    """Format latency with millisecond precision."""
    return f"{ms:.2f} ms"


def format_percentage(val: float) -> str:
    """Format float as percentage string."""
    return f"{val * 100.0:.2f}%"

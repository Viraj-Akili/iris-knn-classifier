"""
Inference Module for IRIS ML PLATFORM.
Real-time tabular species prediction against FastAPI SVM production backend.
"""

import streamlit as st

st.set_page_config(
    page_title="Inference - IRIS ML PLATFORM",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import ApiConnectionError, ApiUnavailableError, ApiValidationError, IrisApiClient
from frontend.components.header import render_header, render_sidebar_connection
from frontend.utils.formatting import apply_custom_theme, format_latency

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_connection(api_client)
render_header(api_client)

st.markdown("## Predict species")
st.caption("Run a real-time prediction against the production SVM model.")

# Presets selector
PRESETS = {
    "Custom Input": (5.8, 2.7, 4.1, 1.0),
    "Typical Setosa (Small petals, wide sepals)": (5.0, 3.5, 1.4, 0.2),
    "Typical Versicolor (Intermediate dimensions)": (5.9, 2.8, 4.2, 1.3),
    "Typical Virginica (Large petals & sepals)": (6.5, 3.0, 5.5, 2.0),
    "Boundary Transition Specimen (Sample #134)": (6.1, 2.6, 5.6, 1.4),
}

selected_preset = st.selectbox(
    "Select Specimen Preset",
    options=list(PRESETS.keys()),
    index=0,
)
default_sl, default_sw, default_pl, default_pw = PRESETS[selected_preset]

st.markdown("### Flower measurements")

# Clean 2x2 grid for numerical inputs
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal length (cm)",
        min_value=0.1,
        max_value=15.0,
        value=float(default_sl),
        step=0.1,
        format="%.1f",
        help="Valid biological range: 0.1 - 15.0 cm",
    )
    petal_length = st.number_input(
        "Petal length (cm)",
        min_value=0.1,
        max_value=15.0,
        value=float(default_pl),
        step=0.1,
        format="%.1f",
        help="Valid biological range: 0.1 - 15.0 cm",
    )

with col2:
    sepal_width = st.number_input(
        "Sepal width (cm)",
        min_value=0.1,
        max_value=15.0,
        value=float(default_sw),
        step=0.1,
        format="%.1f",
        help="Valid biological range: 0.1 - 15.0 cm",
    )
    petal_width = st.number_input(
        "Petal width (cm)",
        min_value=0.1,
        max_value=15.0,
        value=float(default_pw),
        step=0.1,
        format="%.1f",
        help="Valid biological range: 0.1 - 15.0 cm",
    )

st.markdown("<br>", unsafe_allow_html=True)
run_button = st.button("Predict species", type="primary")

if run_button:
    try:
        with st.spinner("Executing inference..."):
            response = api_client.predict(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
            )

        pred_class = response.get("prediction", "Unknown").upper()
        confidence = response.get("confidence", 0.0)
        probabilities = response.get("probabilities", {})
        latency_ms = response.get("inference_latency_ms", 0.0)
        request_id = response.get("request_id", "N/A")
        model_ver = response.get("model_version", "1.0.0")

        # Color class mapping
        color_class = "pred-versicolor"
        if "SETOSA" in pred_class:
            color_class = "pred-setosa"
        elif "VIRGINICA" in pred_class:
            color_class = "pred-virginica"

        # Prediction Result Panel
        st.markdown(
            f"""
            <div class="prediction-panel">
                <div class="prediction-label">Prediction</div>
                <div class="prediction-title {color_class}">{pred_class}</div>
                <div style="font-size: 0.95rem; color: #94a3b8; margin-bottom: 16px;">
                    Confidence: <strong style="color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{confidence * 100:.1f}%</strong>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability distribution bars
        prob_rows = ""
        colors = {"setosa": "#10b981", "versicolor": "#3b82f6", "virginica": "#a855f7"}
        for species_key, prob_val in probabilities.items():
            pct = prob_val * 100.0
            col = colors.get(species_key.lower(), "#3b82f6")
            prob_rows += f"""
            <div class="prob-row">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; width: 100px; color: #94a3b8; font-weight: 600;">{species_key.upper()}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {pct:.1f}%; background-color: {col};"></div>
                </div>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; width: 50px; text-align: right; color: #f8fafc;">{pct:.1f}%</span>
            </div>
            """

        st.markdown(prob_rows, unsafe_allow_html=True)

        st.markdown(
            f"""
                <hr style="margin: 16px 0 12px 0; border-top: 1px solid #334155;"/>
                <div style="display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.78rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                    <span>Inference latency: <strong style="color: #94a3b8;">{format_latency(latency_ms)}</strong></span>
                    <span>•</span>
                    <span>Model: <strong style="color: #94a3b8;">SVM v{model_ver}</strong></span>
                    <span>•</span>
                    <span>Request ID: <strong style="color: #94a3b8;">{request_id[:12]}...</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except ApiValidationError as e:
        st.error(f"Input validation error: {e}")
    except ApiUnavailableError:
        st.error("Production model service is currently unavailable.")
    except ApiConnectionError as e:
        st.error(f"Could not connect to FastAPI backend: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

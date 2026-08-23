"""
IRIS ML - Predict Module
Real-time tabular classification against the production SVM model.
"""

import streamlit as st

st.set_page_config(
    page_title="Predict - IRIS ML",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import ApiConnectionError, ApiUnavailableError, ApiValidationError, IrisApiClient
from frontend.components.header import render_header
from frontend.components.navigation import render_sidebar_navigation
from frontend.utils.formatting import apply_custom_theme, format_latency

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_navigation(api_client)
render_header(api_client, title="Predict a specimen", subtitle="Run a prediction using the production SVM model.")

# Presets selector
PRESETS = {
    "Custom Input": (5.8, 2.7, 4.1, 1.0),
    "Typical Setosa (Small petals, wide sepals)": (5.0, 3.5, 1.4, 0.2),
    "Typical Versicolor (Intermediate dimensions)": (5.9, 2.8, 4.2, 1.3),
    "Typical Virginica (Large petals & sepals)": (6.5, 3.0, 5.5, 2.0),
    "Boundary Transition Specimen (Sample #134)": (6.1, 2.6, 5.6, 1.4),
}

selected_preset = st.selectbox(
    "Specimen Preset",
    options=list(PRESETS.keys()),
    index=0,
)
default_sl, default_sw, default_pl, default_pw = PRESETS[selected_preset]

st.markdown("### Flower measurements")

# Clean 2-column layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal length (cm)",
        min_value=0.1,
        max_value=15.0,
        value=float(default_sl),
        step=0.1,
        format="%.1f",
        help="Valid range: 0.1 - 15.0 cm",
    )
    petal_length = st.number_input(
        "Petal length (cm)",
        min_value=0.1,
        max_value=15.0,
        value=float(default_pl),
        step=0.1,
        format="%.1f",
        help="Valid range: 0.1 - 15.0 cm",
    )

with col2:
    sepal_width = st.number_input(
        "Sepal width (cm)",
        min_value=0.1,
        max_value=15.0,
        value=float(default_sw),
        step=0.1,
        format="%.1f",
        help="Valid range: 0.1 - 15.0 cm",
    )
    petal_width = st.number_input(
        "Petal width (cm)",
        min_value=0.1,
        max_value=15.0,
        value=float(default_pw),
        step=0.1,
        format="%.1f",
        help="Valid range: 0.1 - 15.0 cm",
    )

st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
run_button = st.button("Predict specimen", type="primary")

if run_button:
    try:
        with st.spinner("Classifying specimen..."):
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

        # Result Panel
        st.markdown(
            f"""
            <div class="prediction-panel">
                <div class="prediction-label">Prediction</div>
                <div class="prediction-title {color_class}">{pred_class}</div>
                <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 14px;">
                    Confidence: <strong style="color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{confidence * 100:.2f}%</strong>
                </div>
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; margin-bottom: 8px;">Class Probability</div>
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
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; width: 90px; color: #94a3b8; font-weight: 600;">{species_key.capitalize()}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {pct:.1f}%; background-color: {col};"></div>
                </div>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; width: 55px; text-align: right; color: #f8fafc;">{pct:.2f}%</span>
            </div>
            """

        st.markdown(prob_rows, unsafe_allow_html=True)

        st.markdown(
            f"""
                <hr style="margin: 14px 0 10px 0; border-top: 1px solid #1e293b;"/>
                <div style="display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                    <span>Inference: <strong style="color: #94a3b8;">{format_latency(latency_ms)}</strong></span>
                    <span>•</span>
                    <span>Model: <strong style="color: #94a3b8;">SVM v{model_ver}</strong></span>
                    <span>•</span>
                    <span>Request: <strong style="color: #94a3b8;">{request_id[:8]}...</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View response →"):
            st.json(response)

    except ApiValidationError as e:
        st.error(f"Input validation error: {e}")
    except ApiUnavailableError:
        st.error("Model is currently unavailable.")
    except ApiConnectionError:
        st.error("Unable to reach the inference API.")
    except Exception:
        st.error("Something went wrong while executing the prediction.")

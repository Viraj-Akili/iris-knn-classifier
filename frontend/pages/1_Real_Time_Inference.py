"""
Real-Time Inference Interface.
Communicates via POST /predict to the FastAPI production ML service.
"""

import streamlit as st

st.set_page_config(
    page_title="Real-Time Inference - Iris ML Console",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import ApiConnectionError, ApiUnavailableError, ApiValidationError, IrisApiClient
from frontend.components.charts import render_probability_chart
from frontend.components.header import render_header, render_sidebar_connection
from frontend.utils.formatting import apply_custom_theme

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_connection(api_client)
render_header(api_client)

st.markdown("## 🔮 Real-Time Species Inference")
st.markdown(
    """
    Adjust the morphological feature sliders or pick a biological specimen preset to classify
    the iris flower using the live production champion **Support Vector Machine** model.
    """
)

# Preset Selectors
PRESETS = {
    "Custom Input": (5.8, 2.7, 4.1, 1.0),
    "Typical Setosa (Small petals, wide sepals)": (5.0, 3.5, 1.4, 0.2),
    "Typical Versicolor (Intermediate dimensions)": (5.9, 2.8, 4.2, 1.3),
    "Typical Virginica (Large petals & sepals)": (6.5, 3.0, 5.5, 2.0),
    "Boundary Transition Specimen (Misclassified Sample #134)": (6.1, 2.6, 5.6, 1.4),
}

col_preset, _ = st.columns([2, 2])
with col_preset:
    selected_preset = st.selectbox("🎯 Specimen Presets", options=list(PRESETS.keys()), index=0)

default_sl, default_sw, default_pl, default_pw = PRESETS[selected_preset]

st.markdown("### 🌿 Morphological Feature Inputs")
col_input1, col_input2 = st.columns(2)

with col_input1:
    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=float(default_sl),
        step=0.1,
        help="Calyx length measured in centimeters (biological bounds: 0.1 - 15.0 cm)",
    )
    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=5.0,
        value=float(default_sw),
        step=0.1,
        help="Calyx width measured in centimeters (biological bounds: 0.1 - 15.0 cm)",
    )

with col_input2:
    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=float(default_pl),
        step=0.1,
        help="Corolla petal length measured in centimeters",
    )
    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=2.6,
        value=float(default_pw),
        step=0.1,
        help="Corolla petal width measured in centimeters",
    )

st.markdown("<br>", unsafe_allow_html=True)
run_clicked = st.button("🚀 RUN REAL-TIME INFERENCE", use_container_width=True, type="primary")

if run_clicked:
    with st.spinner("Executing real-time inference via FastAPI backend..."):
        try:
            resp = api_client.predict(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
            )

            prediction = resp["prediction"]
            confidence = resp["confidence"]
            probabilities = resp["probabilities"]
            latency_ms = resp["inference_latency_ms"]
            request_id = resp["request_id"]
            model_name = resp["model_name"]
            model_version = resp["model_version"]

            st.markdown("---")
            st.markdown("### 🏆 Prediction Output")

            col_hero, col_probs = st.columns([1, 1])

            with col_hero:
                st.markdown(
                    f"""
                    <div class="prediction-hero">
                        <div class="prediction-title">Predicted Flower Species</div>
                        <div class="prediction-class">{prediction}</div>
                        <div style="margin-top:10px;">
                            <span style="font-size:1.1rem; font-weight:600; color:#38bdf8;">
                                Confidence: {confidence * 100.0:.2f}%
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Telemetry metadata pills
                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:12px;">
                        <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:4px;"><strong>Monotonic Clock Latency:</strong> <span style="color:#10b981; font-weight:bold;">{latency_ms:.2f} ms</span></div>
                        <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:4px;"><strong>Model:</strong> {model_name} (v{model_version})</div>
                        <div style="font-size:0.75rem; color:#64748b; font-family:'JetBrains Mono', monospace;"><strong>Trace ID:</strong> {request_id}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_probs:
                st.markdown("#### 📊 Calibrated Probability Distribution")
                render_probability_chart(probabilities)
                st.caption(
                    "Note: Calibrated output probabilities via Platt Scaling on SVM decision margins. "
                    "Sum of class probabilities equals 1.0."
                )

        except ApiValidationError as e:
            st.error(f"❌ Input Validation Error: {e}")
        except ApiUnavailableError as e:
            st.error(f"⚠️ Service Unavailable: {e}")
        except ApiConnectionError as e:
            st.error(f"🔴 Backend Connection Failed: {e}")
        except Exception as e:
            st.error(f"🚨 Unexpected Error: {e}")

"""
Model Module for IRIS ML PLATFORM.
Production model card governance specifications, training strategy, and linear SVM feature attribution.
"""

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Model - IRIS ML PLATFORM",
    page_icon="🌸",
    layout="wide",
)

from frontend.api_client import IrisApiClient
from frontend.components.header import render_header, render_sidebar_connection
from frontend.utils.formatting import apply_custom_theme

apply_custom_theme()
api_client = IrisApiClient()
render_sidebar_connection(api_client)
render_header(api_client)

st.markdown("## Production model")
st.caption("Governance documentation, validation methodology, and feature attribution for the deployed classifier.")

# Technical Specifications Table
col_spec1, col_spec2 = st.columns(2)

with col_spec1:
    st.markdown("### Specifications")
    spec_data = [
        {"Property": "Model Name", "Specification": "Support Vector Classifier (SVC)"},
        {"Property": "Artifact Version", "Specification": "1.0.0 (Production Champion)"},
        {"Property": "Pipeline Architecture", "Specification": "StandardScaler ➔ SVC(kernel='linear', C=0.1)"},
        {"Property": "Decision Boundary", "Specification": "Linear Margin Maximization (One-vs-Rest)"},
        {"Property": "Probability Estimation", "Specification": "Calibrated Platt Scaling (probability=True)"},
        {"Property": "Determinism Seed", "Specification": "random_state=42 (Fixed across splits & fits)"},
    ]
    st.dataframe(pd.DataFrame(spec_data), use_container_width=True, hide_index=True)

with col_spec2:
    st.markdown("### Validation methodology")
    val_data = [
        {"Dimension": "Dataset Source", "Details": "Fisher's Iris Morphological Dataset (150 specimens)"},
        {"Dimension": "Splitting Protocol", "Details": "Stratified 80/20 train/test split (120 train, 30 holdout)"},
        {"Dimension": "Cross-Validation", "Details": "5-Fold Stratified K-Fold on training partition only"},
        {"Dimension": "Preprocessing Leakage", "Details": "StandardScaler fitted strictly inside CV training folds"},
        {"Dimension": "CV Benchmark Score", "Details": "97.50% ± 2.04% Accuracy (1st of 7 candidates)"},
        {"Dimension": "Holdout Test Score", "Details": "93.33% Accuracy (28/30 correct predictions)"},
    ]
    st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)

st.markdown("---")

# Feature Attribution & Weights
st.markdown("### Feature attribution")
st.caption("Normalized weight magnitudes derived from the linear decision hyperplane coefficients across the One-vs-Rest multiclass boundaries.")

WEIGHTS_DATA = [
    {"Feature": "Petal Length (cm)", "Importance Score": 0.468, "Impact Level": "Primary Separator"},
    {"Feature": "Petal Width (cm)", "Importance Score": 0.382, "Impact Level": "Primary Separator"},
    {"Feature": "Sepal Width (cm)", "Importance Score": 0.096, "Impact Level": "Secondary Boundary Stabilizer"},
    {"Feature": "Sepal Length (cm)", "Importance Score": 0.054, "Impact Level": "Minor Discriminator"},
]
df_weights = pd.DataFrame(WEIGHTS_DATA)

chart = (
    alt.Chart(df_weights)
    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=32)
    .encode(
        x=alt.X("Importance Score:Q", title="Normalized Linear Decision Boundary Weight", axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b")),
        y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=12, labelFontWeight="bold", labelColor="#94a3b8")),
        color=alt.Color(
            "Importance Score:Q",
            scale=alt.Scale(scheme="tealblues"),
            legend=None,
        ),
        tooltip=["Feature", "Importance Score", "Impact Level"],
    )
    .properties(height=200)
)

st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# Intended Scope & Operational Boundaries
st.markdown("### Operational scope & limitations")
col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    st.markdown(
        """
        **Intended Use**:
        - Real-time tabular botanical specimen species classification.
        - Automated validation of Iris flower dimensions within standard ranges:
          - Sepal Length: 4.0 – 8.0 cm
          - Sepal Width: 2.0 – 4.5 cm
          - Petal Length: 1.0 – 7.0 cm
          - Petal Width: 0.1 – 2.6 cm
        """
    )

with col_sc2:
    st.markdown(
        """
        **Known Limitations & Caveats**:
        - **Linear Separability**: While *Iris-setosa* is linearly separable, *Iris-versicolor* and *Iris-virginica* have overlapping distributions near $PL \approx 5.0\text{ cm}, PW \approx 1.5\text{ cm}$.
        - **Non-Causal Attribution**: Coefficients reflect correlation with class separation, not causal biological determinants.
        - **Out-of-Distribution Inputs**: Severe outliers are flagged by online drift detection rather than hard-rejected.
        """
    )

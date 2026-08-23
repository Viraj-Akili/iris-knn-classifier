"""
Model Tournament & Cross-Validation Benchmarking Page.
Displays the 7-algorithm ML tournament results and diagnostic plots from Phase 1.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Model Tournament - Iris ML Console",
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

st.markdown("## 🏆 7-Model Benchmark Tournament Leaderboard")
st.markdown(
    """
    Review the systematic 5-fold cross-validation benchmarking tournament evaluating
    7 candidate machine learning algorithms under rigorous leakage-free preprocessing.
    """
)

# Load persisted comparison CSV
base_dir = Path(__file__).resolve().parent.parent.parent
csv_path = base_dir / "artifacts" / "experiments" / "model_comparison.csv"

if csv_path.exists():
    df = pd.read_csv(csv_path)

    st.markdown("### 🥇 Multi-Model Tournament Results (Training Folds N=120)")

    # Format and display table
    display_df = df.copy()
    if "cv_accuracy_mean" in display_df.columns:
        display_df["CV Accuracy"] = display_df.apply(
            lambda r: f"{r['cv_accuracy_mean'] * 100:.2f}% ± {r['cv_accuracy_std'] * 100:.2f}%", axis=1
        )
    if "cv_f1_macro" in display_df.columns:
        display_df["CV Macro F1"] = display_df["cv_f1_macro"].apply(lambda v: f"{v:.4f}")
    if "cv_f1_weighted" in display_df.columns:
        display_df["CV Weighted F1"] = display_df["cv_f1_weighted"].apply(lambda v: f"{v:.4f}")
    if "test_accuracy" in display_df.columns:
        display_df["Holdout Test Acc"] = display_df["test_accuracy"].apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) and v > 0 else "Untouched"
        )

    cols_to_show = [
        col for col in ["rank", "model_name", "CV Accuracy", "CV Macro F1", "CV Weighted F1", "Holdout Test Acc", "best_params"]
        if col in display_df.columns
    ]
    st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True)

    st.markdown(
        r"""
        > 💡 **Champion Selection Rationale**: **Support Vector Machine (Linear Kernel, C=0.1)** achieved the highest
        > cross-validation accuracy (**97.50%**) and lowest variance across folds ($\pm 2.04\%$), outperforming distance-based
        > and tree-ensemble classifiers while offering superior margin stability.
        """
    )
else:
    st.warning("Model comparison artifact `artifacts/experiments/model_comparison.csv` not found.")

st.markdown("---")
st.markdown("### 📈 Visual Benchmark Diagnostics")

plot_comp_path = base_dir / "artifacts" / "plots" / "model_comparison.png"
plot_cv_path = base_dir / "artifacts" / "plots" / "cv_performance.png"

col_p1, col_p2 = st.columns(2)

with col_p1:
    if plot_comp_path.exists():
        st.image(str(plot_comp_path), caption="Figure 1: Cross-Validation Accuracy & Macro F1 Ranking", use_container_width=True)
    else:
        st.info("Model comparison plot not found.")

with col_p2:
    if plot_cv_path.exists():
        st.image(str(plot_cv_path), caption="Figure 2: 5-Fold Cross-Validation Variance Analysis (±1σ)", use_container_width=True)
    else:
        st.info("CV performance plot not found.")

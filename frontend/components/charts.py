"""
Interactive chart generators using Altair, Pandas, and Streamlit native chart components.
"""

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


def render_probability_chart(probabilities: dict[str, float]) -> None:
    """Render horizontal bar chart for predicted class probabilities with custom color mapping."""
    df = pd.DataFrame([
        {"Species": k.capitalize(), "Probability": v, "Percentage": f"{v * 100.0:.2f}%"}
        for k, v in probabilities.items()
    ])

    color_scale = alt.Scale(
        domain=["Setosa", "Versicolor", "Virginica"],
        range=["#22c55e", "#3b82f6", "#a855f7"],
    )

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, height=32)
        .encode(
            x=alt.X(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="Calibrated Class Probability"),
            ),
            y=alt.Y("Species:N", sort=None, title=None, axis=alt.Axis(labelFontSize=13, labelFontWeight="bold")),
            color=alt.Color("Species:N", scale=color_scale, legend=None),
            tooltip=["Species", "Percentage"],
        )
        .properties(height=160)
    )

    text = chart.mark_text(
        align="left",
        baseline="middle",
        dx=8,
        fontSize=12,
        fontWeight="bold",
        color="#f8fafc",
    ).encode(text="Percentage:N")

    st.altair_chart(chart + text, use_container_width=True)


def render_class_distribution_chart(class_counts: dict[str, int]) -> None:
    """Render bar chart of runtime prediction counts per class."""
    if not class_counts or sum(class_counts.values()) == 0:
        st.info("No runtime predictions recorded yet. Run inferences to populate chart.")
        return

    df = pd.DataFrame([
        {"Species": k.capitalize(), "Predictions": v}
        for k, v in class_counts.items()
    ])

    color_scale = alt.Scale(
        domain=["Setosa", "Versicolor", "Virginica"],
        range=["#22c55e", "#3b82f6", "#a855f7"],
    )

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=6)
        .encode(
            x=alt.X("Species:N", title="Flower Species", axis=alt.Axis(labelFontSize=12)),
            y=alt.Y("Predictions:Q", title="Total Prediction Count"),
            color=alt.Color("Species:N", scale=color_scale, legend=None),
            tooltip=["Species", "Predictions"],
        )
        .properties(height=240)
    )

    text = chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-4,
        fontSize=12,
        fontWeight="bold",
        color="#f8fafc",
    ).encode(text="Predictions:Q")

    st.altair_chart(chart + text, use_container_width=True)


def render_confidence_chart(confidence_dist: dict[str, int]) -> None:
    """Render confidence tier distribution (High, Medium, Low)."""
    if not confidence_dist or sum(confidence_dist.values()) == 0:
        st.info("No predictions recorded yet.")
        return

    df = pd.DataFrame([
        {"Tier": "High (>= 90%)", "Count": confidence_dist.get("high", 0), "Color": "#10b981"},
        {"Tier": "Medium (70-90%)", "Count": confidence_dist.get("medium", 0), "Color": "#f59e0b"},
        {"Tier": "Low (< 70%)", "Count": confidence_dist.get("low", 0), "Color": "#ef4444"},
    ])

    color_scale = alt.Scale(
        domain=["High (>= 90%)", "Medium (70-90%)", "Low (< 70%)"],
        range=["#10b981", "#f59e0b", "#ef4444"],
    )

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=6)
        .encode(
            x=alt.X("Tier:N", title="Confidence Tier", sort=None),
            y=alt.Y("Count:Q", title="Observations"),
            color=alt.Color("Tier:N", scale=color_scale, legend=None),
            tooltip=["Tier", "Count"],
        )
        .properties(height=240)
    )

    text = chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-4,
        fontSize=12,
        fontWeight="bold",
        color="#f8fafc",
    ).encode(text="Count:Q")

    st.altair_chart(chart + text, use_container_width=True)


def render_feature_distribution_chart(feature_stats: dict[str, dict[str, Any]]) -> None:
    """Render comparison chart of online feature means and standard deviations."""
    if not feature_stats:
        return

    records = []
    for k, v in feature_stats.items():
        if v.get("count", 0) > 0:
            records.append({
                "Feature": k.replace("_", " ").title() + " (cm)",
                "Mean": v.get("mean", 0.0),
                "Std": v.get("std", 0.0),
                "Min": v.get("min", 0.0),
                "Max": v.get("max", 0.0),
            })

    if not records:
        st.info("No incoming feature statistics recorded yet.")
        return

    df = pd.DataFrame(records)

    base = alt.Chart(df).encode(
        y=alt.Y("Feature:N", title=None, sort=None)
    )

    bars = base.mark_bar(color="#6366f1", opacity=0.7).encode(
        x=alt.X("Mean:Q", title="Feature Measurement Mean & Observed Range (cm)")
    )

    ticks = base.mark_tick(color="#f8fafc", thickness=3).encode(
        x="Mean:Q"
    )

    st.altair_chart((bars + ticks).properties(height=200), use_container_width=True)

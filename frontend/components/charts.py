"""
Clean, restrained chart generators using Altair and Streamlit native components.
"""

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


def render_probability_chart(probabilities: dict[str, float]) -> None:
    """Render horizontal bar chart for predicted class probabilities."""
    df = pd.DataFrame([
        {"Species": k.upper(), "Probability": v, "Percentage": f"{v * 100.0:.1f}%"}
        for k, v in probabilities.items()
    ])

    color_scale = alt.Scale(
        domain=["SETOSA", "VERSICOLOR", "VIRGINICA"],
        range=["#10b981", "#3b82f6", "#a855f7"],
    )

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=26)
        .encode(
            x=alt.X(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title=None, labels=False, ticks=False),
            ),
            y=alt.Y("Species:N", sort=None, title=None, axis=alt.Axis(labelFontSize=12, labelFontWeight="bold", labelColor="#94a3b8")),
            color=alt.Color("Species:N", scale=color_scale, legend=None),
            tooltip=["Species", "Percentage"],
        )
        .properties(height=130)
    )

    text = chart.mark_text(
        align="left",
        baseline="middle",
        dx=6,
        fontSize=11,
        fontWeight="bold",
        color="#f8fafc",
    ).encode(text="Percentage:N")

    st.altair_chart(chart + text, use_container_width=True)


def render_class_distribution_chart(class_counts: dict[str, int]) -> None:
    """Render clean bar chart of runtime prediction counts per class."""
    if not class_counts or sum(class_counts.values()) == 0:
        st.caption("No prediction events recorded yet.")
        return

    df = pd.DataFrame([
        {"Species": k.upper(), "Predictions": v}
        for k, v in class_counts.items()
    ])

    color_scale = alt.Scale(
        domain=["SETOSA", "VERSICOLOR", "VIRGINICA"],
        range=["#10b981", "#3b82f6", "#a855f7"],
    )

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("Species:N", title=None, axis=alt.Axis(labelFontSize=11, labelColor="#94a3b8")),
            y=alt.Y("Predictions:Q", title="Total Inferences", axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b")),
            color=alt.Color("Species:N", scale=color_scale, legend=None),
            tooltip=["Species", "Predictions"],
        )
        .properties(height=200)
    )

    text = chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-3,
        fontSize=11,
        fontWeight="bold",
        color="#f8fafc",
    ).encode(text="Predictions:Q")

    st.altair_chart(chart + text, use_container_width=True)


def render_confidence_chart(confidence_dist: dict[str, int]) -> None:
    """Render confidence tier distribution (High, Medium, Low)."""
    if not confidence_dist or sum(confidence_dist.values()) == 0:
        st.caption("No confidence observations recorded yet.")
        return

    df = pd.DataFrame([
        {"Tier": "High (>=90%)", "Count": confidence_dist.get("high", 0)},
        {"Tier": "Medium (70-90%)", "Count": confidence_dist.get("medium", 0)},
        {"Tier": "Low (<70%)", "Count": confidence_dist.get("low", 0)},
    ])

    color_scale = alt.Scale(
        domain=["High (>=90%)", "Medium (70-90%)", "Low (<70%)"],
        range=["#10b981", "#f59e0b", "#ef4444"],
    )

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("Tier:N", title=None, sort=None, axis=alt.Axis(labelFontSize=11, labelColor="#94a3b8")),
            y=alt.Y("Count:Q", title="Observations", axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b")),
            color=alt.Color("Tier:N", scale=color_scale, legend=None),
            tooltip=["Tier", "Count"],
        )
        .properties(height=200)
    )

    text = chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-3,
        fontSize=11,
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
        st.caption("No incoming feature measurements accumulated yet.")
        return

    df = pd.DataFrame(records)

    base = alt.Chart(df).encode(
        y=alt.Y("Feature:N", title=None, sort=None, axis=alt.Axis(labelColor="#94a3b8"))
    )

    bars = base.mark_bar(color="#3b82f6", opacity=0.6, cornerRadius=4).encode(
        x=alt.X("Mean:Q", title="Observed Mean (cm)", axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b"))
    )

    ticks = base.mark_tick(color="#f8fafc", thickness=2).encode(
        x="Mean:Q"
    )

    st.altair_chart((bars + ticks).properties(height=180), use_container_width=True)

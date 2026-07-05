"""
ui/ats_page.py
---------------
ATS Score page.

Responsibilities:
    - Compute (or reuse a cached) ATS score via core.ats_scorer
      .calculate_ats_score(), using the flat parsed resume and the
      optional job description text.
    - Display the overall score with a band label (Excellent / Good /
      Fair / Weak) from config.ATS_SCORE_BANDS.
    - Display a Plotly bar chart of the per-category breakdown
      (skills / experience / projects / education / structure) against
      their maximum possible points (config.ATS_WEIGHTS).
    - List missing keywords (if a job description was provided).

Notes:
    - The ATS score is JD-independent for the "structure" component
      and partially JD-dependent for the others (see core.ats_scorer
      docstring). If no JD is provided, JD-dependent categories
      default to full marks.
    - Results are cached in session_state (ats.result) and only
      recomputed when the user clicks "Calculate ATS Score", or
      automatically on first visit if no cached result exists.
"""

from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go
import streamlit as st

from config import ATS_WEIGHTS, ATS_SCORE_BANDS, ATS_BAND_COLORS, CHART_PRIMARY_COLOR
from core.ats_scorer import calculate_ats_score
from utils.session_state import (
    require_resume,
    get_parsed_resume,
    get_job_description,
    get_ats_result,
    set_ats_result,
    jd_is_provided,
)


def _score_band(overall_score: int) -> str:
    """
    Map an overall ATS score (0-100) to a band label.

    Args:
        overall_score: The overall ATS score.

    Returns:
        One of the keys in config.ATS_SCORE_BANDS (e.g. "Excellent",
        "Good", "Fair", "Weak"). Falls back to "Weak" if no band's
        range matches (should not normally happen for a 0-100 score).
    """
    for band_name, (low, high) in ATS_SCORE_BANDS.items():
        if low <= overall_score <= high:
            return band_name
    return "Weak"


def _render_score_gauge(overall_score: int, band: str) -> None:
    """
    Render a Plotly gauge chart for the overall ATS score.

    Args:
        overall_score: The overall ATS score (0-100).
        band: The score band label (used to pick the gauge color).
    """
    band_color = ATS_BAND_COLORS.get(band, CHART_PRIMARY_COLOR)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=overall_score,
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": band_color},
                "steps": [
                    {"range": [low, high], "color": "rgba(0,0,0,0.05)"}
                    for _, (low, high) in ATS_SCORE_BANDS.items()
                ],
            },
            title={"text": f"Overall ATS Score — {band}"},
        )
    )
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


def _render_breakdown_chart(result: Dict) -> None:
    """
    Render a Plotly bar chart comparing each category's score against
    its maximum possible points.

    Args:
        result: The ATS result dict from calculate_ats_score().
    """
    categories = ["skills", "experience", "projects", "education", "structure"]
    labels = [c.capitalize() for c in categories]

    scores = [result.get(f"{c}_score", 0) for c in categories]
    max_points = [ATS_WEIGHTS.get(c, 0) for c in categories]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Your score",
            x=labels,
            y=scores,
            marker_color=CHART_PRIMARY_COLOR,
            text=[f"{s:.1f}" for s in scores],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Maximum possible",
            x=labels,
            y=max_points,
            marker_color="rgba(0,0,0,0.1)",
        )
    )

    fig.update_layout(
        barmode="overlay",
        height=350,
        margin=dict(t=30, b=20, l=20, r=20),
        yaxis_title="Points",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_missing_keywords(result: Dict) -> None:
    """
    Render the list of JD keywords not found in the resume, if a job
    description was provided and any are missing.

    Args:
        result: The ATS result dict from calculate_ats_score().
    """
    missing = result.get("missing_keywords", [])

    if result.get("no_jd", False) or not jd_is_provided():
        st.warning(
            "⚠️ No job description provided. The score above only reflects "
            "your resume structure (max 10 points). "
            "Go to the **JD Match** page, paste a job description, then "
            "return here and click **Recalculate ATS Score** for a real score.",
            icon="📋",
        )
        return

    st.subheader("Missing Keywords")

    if not missing:
        st.success("No missing keywords detected — great match!", icon="✅")
        return

    st.caption(
        f"{len(missing)} keyword(s) from the job description were not "
        "found anywhere in your resume:"
    )
    # Render as wrapped "chips" using columns of markdown badges.
    chip_text = "  ".join(f"`{kw}`" for kw in missing)
    st.markdown(chip_text)


def render() -> None:
    """
    Render the ATS Score page.

    Computes the ATS score on first visit (or on demand via the
    "Recalculate" button), then displays a gauge, a category
    breakdown chart, and any missing keywords.
    """
    if not require_resume("ATS Score"):
        return

    parsed_resume = get_parsed_resume()
    job_description = get_job_description()

    cached_result = get_ats_result()

    recalc = st.button("🔄 Recalculate ATS Score", use_container_width=False)

    if cached_result is None or recalc:
        with st.spinner("Calculating ATS score..."):
            result = calculate_ats_score(parsed_resume, job_description)
        set_ats_result(result)
    else:
        result = cached_result

    overall_score = int(result.get("overall_score", 0))
    band = _score_band(overall_score)

    _render_score_gauge(overall_score, band)

    st.subheader("Score Breakdown")
    st.caption(
        "Each category is scored out of its maximum points "
        f"(Skills {ATS_WEIGHTS['skills']}, Experience {ATS_WEIGHTS['experience']}, "
        f"Projects {ATS_WEIGHTS['projects']}, Education {ATS_WEIGHTS['education']}, "
        f"Structure {ATS_WEIGHTS['structure']})."
    )
    _render_breakdown_chart(result)

    st.divider()
    _render_missing_keywords(result)
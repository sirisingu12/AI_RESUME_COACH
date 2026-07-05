"""
ui/jd_match_page.py
---------------------
Job Description Match page.

Responsibilities:
    - Provide a text area for the user to paste a target job
      description (stored in session_state via set_job_description()).
    - Compute semantic match score + matched/missing skills via
      core.jd_matcher.match_resume_to_job() (sentence-transformers).
    - Compute keyword-level present/missing analysis via
      core.keyword_gap.analyze_keyword_gap() (pure regex, no LLM).
    - Display both results: a match-score gauge, matched/missing skill
      chips, and a keyword coverage bar.

Notes:
    - Both core.jd_matcher and core.keyword_gap independently extract
      keywords using similar (but not identical) heuristics, so their
      "matched"/"present" lists may differ slightly — this is
      expected and both are shown for a fuller picture.
    - Entering/changing the JD via set_job_description() automatically
      invalidates cached match/gap/ATS results (see
      utils.session_state.set_job_description), so this page always
      recomputes against the latest JD text.
"""

from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go
import streamlit as st

from config import CHART_PRIMARY_COLOR, CHART_COLOR_SEQUENCE
from core.jd_matcher import match_resume_to_job
from core.keyword_gap import analyze_keyword_gap
from utils.session_state import (
    require_resume,
    get_resume_text,
    get_job_description,
    set_job_description,
    get_match_result,
    set_match_result,
    get_gap_result,
    set_gap_result,
    jd_is_provided,
)
from utils.text_cleaning import is_meaningful_text


def _render_match_gauge(match_score: float) -> None:
    """
    Render a Plotly gauge chart for the semantic match score.

    Args:
        match_score: Semantic similarity score (0-100).
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=match_score,
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": CHART_PRIMARY_COLOR},
            },
            title={"text": "Semantic Match Score"},
        )
    )
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


def _render_keyword_coverage_bar(gap_result: Dict) -> None:
    """
    Render a horizontal stacked bar showing the proportion of JD
    keywords present vs. missing in the resume.

    Args:
        gap_result: The result dict from analyze_keyword_gap().
    """
    present = len(gap_result.get("present_keywords", []))
    missing = len(gap_result.get("missing_keywords", []))
    total = gap_result.get("total_keywords", present + missing)

    if total == 0:
        st.caption("No keywords could be extracted from the job description.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Keyword coverage"],
            x=[present],
            name="Present",
            orientation="h",
            marker_color=CHART_COLOR_SEQUENCE[1],
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Keyword coverage"],
            x=[missing],
            name="Missing",
            orientation="h",
            marker_color=CHART_COLOR_SEQUENCE[2],
        )
    )
    fig.update_layout(
        barmode="stack",
        height=140,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_title=f"Keywords (out of {total})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    match_pct = gap_result.get("match_percentage", 0.0)
    st.caption(f"**{present}/{total}** keywords found ({match_pct}% coverage).")


def _render_skill_chips(title: str, skills: list, empty_message: str) -> None:
    """
    Render a list of skills as inline markdown code chips.

    Args:
        title: Section heading.
        skills: List of skill/keyword strings.
        empty_message: Message shown if `skills` is empty.
    """
    st.markdown(f"**{title}**")
    if not skills:
        st.caption(empty_message)
        return
    chip_text = "  ".join(f"`{s}`" for s in skills)
    st.markdown(chip_text)


def _compute_results(resume_text: str, job_description: str) -> None:
    """
    Run jd_matcher and keyword_gap analyses and store results in
    session_state.

    Args:
        resume_text: The full raw resume text.
        job_description: The job description text.
    """
    with st.spinner("Analyzing semantic match (this may take a moment)..."):
        match_result = match_resume_to_job(resume_text, job_description)
    set_match_result(match_result)

    with st.spinner("Analyzing keyword overlap..."):
        gap_result = analyze_keyword_gap(resume_text, job_description)
    set_gap_result(gap_result)


def render() -> None:
    """
    Render the Job Description Match page.

    Workflow:
        1. Show a text area for the job description.
        2. On submit (or if no cached result exists and a JD is
           present), compute the semantic match and keyword gap.
        3. Display the match score gauge, matched/missing skill chips,
           and the keyword coverage bar with missing keyword chips.
    """
    if not require_resume("JD Match"):
        return

    st.write(
        "Paste the job description you're targeting below. Your resume "
        "will be compared against it using local semantic embeddings "
        "(no data leaves your machine)."
    )

    jd_text = st.text_area(
        "Job description",
        value=get_job_description(),
        height=220,
        placeholder="Paste the full job description here...",
    )

    col1, col2 = st.columns([1, 3])
    submitted = col1.button("🔍 Analyze Match", type="primary")

    if submitted:
        if not is_meaningful_text(jd_text, min_words=10):
            st.warning(
                "Please paste a more complete job description "
                "(at least a few sentences) for a meaningful analysis.",
                icon="⚠️",
            )
        else:
            set_job_description(jd_text)  # also invalidates cached results
            _compute_results(get_resume_text(), jd_text)

    st.divider()

    if not jd_is_provided():
        st.caption("Enter a job description above and click **Analyze Match** to see results.")
        return

    match_result = get_match_result()
    gap_result = get_gap_result()

    if match_result is None or gap_result is None:
        st.caption("Click **Analyze Match** to compare your resume against this job description.")
        return

    # --- Semantic match -----------------------------------------------
    _render_match_gauge(match_result.get("match_score", 0.0))

    col1, col2 = st.columns(2)
    with col1:
        _render_skill_chips(
            "Matched skills (semantic)",
            match_result.get("matched_skills", []),
            "No matched skills detected.",
        )
    with col2:
        _render_skill_chips(
            "Missing skills (semantic)",
            match_result.get("missing_skills", []),
            "No missing skills detected — great coverage!",
        )

    st.divider()

    # --- Keyword gap ----------------------------------------------------
    st.subheader("Keyword Coverage")
    _render_keyword_coverage_bar(gap_result)

    col1, col2 = st.columns(2)
    with col1:
        _render_skill_chips(
            "Present keywords",
            gap_result.get("present_keywords", []),
            "No keywords matched.",
        )
    with col2:
        _render_skill_chips(
            "Missing keywords",
            gap_result.get("missing_keywords", []),
            "No missing keywords — great coverage!",
        )

"""
ui/roadmap_page.py
--------------------
Career Roadmap Generator page.

Responsibilities:
    - Accept an optional target role from the user.
    - Call core.career_roadmap.generate_roadmap() (LM Studio LLM) with
      the resume text and target role.
    - Display the structured result: skill gaps, recommended
      technologies, a staged learning roadmap (rendered as a Plotly
      Gantt-style timeline), and suggested projects.
    - Cache the result in session_state (roadmap.result); provide a
      "Regenerate" control.

Notes:
    - Mirrors the error-handling pattern used elsewhere: errors are
      encoded as a single message inside "skill_gaps" while all other
      lists are empty.
    - The "duration" field on each learning_roadmap stage is a
      free-form string from the LLM (e.g. "4-6 weeks", "1 month").
      _parse_duration_weeks() attempts to convert this to an
      approximate number of weeks for the timeline; stages whose
      duration cannot be parsed are given a default width.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import plotly.express as px
import streamlit as st
import pandas as pd

from config import CHART_COLOR_SEQUENCE
from core.career_roadmap import generate_roadmap
from utils.session_state import (
    require_resume,
    get_resume_text,
    get_roadmap_result,
    set_roadmap_result,
)
from utils.text_cleaning import format_list_as_bullets


# Default duration (in weeks) used for stages whose "duration" string
# cannot be parsed into a number.
DEFAULT_STAGE_WEEKS: float = 4.0


def _is_error_result(result: Dict) -> Optional[str]:
    """
    Check whether a roadmap result represents an LLM/parsing error
    rather than a real roadmap.

    Args:
        result: The roadmap result dict.

    Returns:
        The error message string if this result represents an error,
        or None if it looks like a normal roadmap.
    """
    skill_gaps = result.get("skill_gaps", [])
    other_lists = (
        result.get("recommended_technologies", [])
        + result.get("learning_roadmap", [])
        + result.get("suggested_projects", [])
    )

    if len(skill_gaps) == 1 and not other_lists:
        message = skill_gaps[0]
        if message.startswith("[LLM Error]") or "could not be parsed" in message.lower() or "No resume text" in message:
            return message

    return None


def _parse_duration_weeks(duration: str) -> float:
    """
    Estimate a number of weeks from a free-form duration string.

    Handles common patterns like "4 weeks", "4-6 weeks", "1 month",
    "2 months", "1-2 months". For ranges, the average of the bounds is
    used. Falls back to DEFAULT_STAGE_WEEKS if nothing can be parsed.

    Args:
        duration: A free-form duration string from the LLM.

    Returns:
        An estimated duration in weeks (always > 0).
    """
    if not duration:
        return DEFAULT_STAGE_WEEKS

    text = duration.lower()

    # Match one or two numbers (for ranges like "4-6").
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return DEFAULT_STAGE_WEEKS

    value = sum(numbers) / len(numbers)

    if "month" in text:
        value *= 4.345  # approx weeks per month
    elif "day" in text:
        value /= 7.0
    # default unit: weeks (covers "week"/"weeks" and unlabeled numbers)

    return max(value, 0.5)


def _run_generation(target_role: str) -> None:
    """
    Call generate_roadmap() with the current resume text and target
    role, and store the result in session_state.

    Args:
        target_role: The user-provided target role (may be empty).
    """
    resume_text = get_resume_text()

    with st.spinner("Generating your career roadmap... this may take a minute."):
        result = generate_roadmap(resume_text, target_role=target_role)

    set_roadmap_result(result)


def _render_timeline(stages: List[Dict]) -> None:
    """
    Render the learning roadmap stages as a Plotly horizontal timeline
    (Gantt-style), using parsed durations to size each bar.

    Args:
        stages: List of stage dicts, each with "stage" (str),
                "duration" (str), and "topics" (List[str]).
    """
    if not stages:
        st.caption("No roadmap stages were generated.")
        return

    rows = []
    cursor_weeks = 0.0

    for stage in stages:
        weeks = _parse_duration_weeks(stage.get("duration", ""))
        start = cursor_weeks
        end = cursor_weeks + weeks
        cursor_weeks = end

        topics = stage.get("topics", [])
        topics_str = ", ".join(topics) if topics else "—"

        rows.append(
            {
                "Stage": stage.get("stage") or "Stage",
                "Start": start,
                "End": end,
                "Duration": stage.get("duration") or "Unspecified",
                "Topics": topics_str,
            }
        )

    df = pd.DataFrame(rows)

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Stage",
        color="Stage",
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
        hover_data={"Duration": True, "Topics": True, "Start": False, "End": False},
    )

    # px.timeline expects datetime axes by default; since we're using
    # plain numeric "weeks from start", override the x-axis type.
    fig.update_xaxes(type="linear", title="Weeks from start")
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(height=80 + 60 * len(rows), showlegend=False, margin=dict(t=20, b=20, l=20, r=20))

    st.plotly_chart(fig, use_container_width=True)


def _render_stage_details(stages: List[Dict]) -> None:
    """
    Render a detail expander per stage, listing its topics.

    Args:
        stages: List of stage dicts, each with "stage", "duration",
                and "topics".
    """
    for i, stage in enumerate(stages, start=1):
        title = stage.get("stage") or f"Stage {i}"
        duration = stage.get("duration") or "Unspecified duration"
        topics = stage.get("topics", [])

        with st.expander(f"{title} ({duration})"):
            if topics:
                st.markdown(format_list_as_bullets(topics))
            else:
                st.caption("No specific topics listed for this stage.")


def render() -> None:
    """
    Render the Career Roadmap Generator page.

    Workflow:
        1. Optional text input for a target role.
        2. Generate (or regenerate) the roadmap via the local LLM.
        3. Display skill gaps, recommended technologies, a Plotly
           timeline of learning stages, stage details, and suggested
           projects.
    """
    if not require_resume("Career Roadmap"):
        return

    st.write(
        "Generate a personalized learning roadmap based on your resume "
        "and (optionally) a target role you're aiming for."
    )

    target_role = st.text_input(
        "Target role (optional)",
        placeholder="e.g. Machine Learning Engineer, Backend Developer, Data Analyst",
    )

    cached_result = get_roadmap_result()

    col1, _ = st.columns([1, 3])
    regenerate = col1.button(
        "🔄 Regenerate" if cached_result is not None else "✨ Generate Roadmap",
        type="primary",
    )

    if cached_result is None or regenerate:
        _run_generation(target_role)

    result = get_roadmap_result()
    if result is None:
        return

    st.divider()

    error_message = _is_error_result(result)
    if error_message:
        st.error(error_message, icon="⚠️")
        st.caption(
            "Make sure LM Studio is running with a model loaded, then "
            "click **Regenerate** above."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧩 Skill Gaps")
        items = result.get("skill_gaps", [])
        if items:
            st.markdown(format_list_as_bullets(items))
        else:
            st.caption("No skill gaps identified.")

    with col2:
        st.subheader("🛠️ Recommended Technologies")
        items = result.get("recommended_technologies", [])
        if items:
            st.markdown(format_list_as_bullets(items))
        else:
            st.caption("No specific technologies recommended.")

    st.divider()

    st.subheader("🗺️ Learning Roadmap")
    stages = result.get("learning_roadmap", [])
    _render_timeline(stages)
    _render_stage_details(stages)

    st.divider()

    st.subheader("🚀 Suggested Projects")
    items = result.get("suggested_projects", [])
    if items:
        st.markdown(format_list_as_bullets(items))
    else:
        st.caption("No specific projects suggested.")

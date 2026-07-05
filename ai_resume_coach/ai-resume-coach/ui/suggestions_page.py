"""
ui/suggestions_page.py
------------------------
Resume Improvement Suggestions page.

Responsibilities:
    - Call core.suggestion_engine.review_resume() (LM Studio LLM) with
      the resume text and optional job description.
    - Display the structured result: strengths, weaknesses, missing
      sections, and improvement suggestions.
    - Cache the result in session_state (review.result) so revisiting
      this page doesn't re-trigger an LLM call; provide a "Regenerate"
      button for an explicit re-run.

Notes:
    - This is the first LLM-powered page in the app. If LM Studio is
      not running, core.suggestion_engine.review_resume() returns a
      result where "weaknesses" contains a single "[LLM Error] ..."
      string — this page detects and displays that as an error rather
      than rendering it as a normal weakness.
"""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

from core.suggestion_engine import review_resume
from utils.session_state import (
    require_resume,
    get_resume_text,
    get_job_description,
    get_review_result,
    set_review_result,
    jd_is_provided,
)
from utils.text_cleaning import format_list_as_bullets


def _is_error_result(result: Dict) -> str | None:
    """
    Check whether a review result represents an LLM/parsing error
    rather than a real review.

    core.suggestion_engine.review_resume() encodes errors as a single
    "[LLM Error] ..." or similar message inside "weaknesses" while
    leaving all other lists empty.

    Args:
        result: The review result dict.

    Returns:
        The error message string if this result represents an error,
        or None if it looks like a normal review.
    """
    weaknesses = result.get("weaknesses", [])
    other_lists = (
        result.get("strengths", [])
        + result.get("missing_sections", [])
        + result.get("improvement_suggestions", [])
    )

    if len(weaknesses) == 1 and not other_lists:
        message = weaknesses[0]
        if message.startswith("[LLM Error]") or "could not be parsed" in message.lower() or "No resume text" in message:
            return message

    return None


def _run_review() -> None:
    """
    Call review_resume() with the current resume text and job
    description, and store the result in session_state.
    """
    resume_text = get_resume_text()
    job_description = get_job_description()

    with st.spinner("Asking the local AI to review your resume... this may take a minute."):
        result = review_resume(resume_text, job_description)

    set_review_result(result)


def _render_result(result: Dict) -> None:
    """
    Render a successful review result as four sections.

    Args:
        result: The review result dict with "strengths", "weaknesses",
                "missing_sections", and "improvement_suggestions".
    """
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Strengths")
        items = result.get("strengths", [])
        if items:
            st.markdown(format_list_as_bullets(items))
        else:
            st.caption("No specific strengths identified.")

        st.subheader("📋 Missing Sections")
        items = result.get("missing_sections", [])
        if items:
            st.markdown(format_list_as_bullets(items))
        else:
            st.caption("No missing sections identified — your resume looks complete.")

    with col2:
        st.subheader("⚠️ Weaknesses")
        items = result.get("weaknesses", [])
        if items:
            st.markdown(format_list_as_bullets(items))
        else:
            st.caption("No specific weaknesses identified.")

        st.subheader("💡 Improvement Suggestions")
        items = result.get("improvement_suggestions", [])
        if items:
            st.markdown(format_list_as_bullets(items))
        else:
            st.caption("No specific suggestions.")


def render() -> None:
    """
    Render the Resume Improvement Suggestions page.

    On first visit (no cached result), automatically runs the review.
    A "Regenerate" button allows re-running against updated resume/JD
    text.
    """
    if not require_resume("Suggestions"):
        return

    st.write(
        "Get AI-powered feedback on your resume's strengths, "
        "weaknesses, missing sections, and concrete improvement ideas."
    )

    if jd_is_provided():
        st.caption("A job description is set — feedback will be tailored to it where relevant.")
    else:
        st.caption(
            "No job description set. For more tailored feedback, add one "
            "on the **JD Match** page first."
        )

    cached_result = get_review_result()

    col1, _ = st.columns([1, 3])
    regenerate = col1.button(
        "🔄 Regenerate" if cached_result is not None else "✨ Generate Suggestions",
        type="primary",
    )

    if cached_result is None or regenerate:
        _run_review()

    result = get_review_result()
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

    _render_result(result)

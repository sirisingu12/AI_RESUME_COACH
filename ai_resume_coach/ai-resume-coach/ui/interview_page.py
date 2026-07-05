"""
ui/interview_page.py
----------------------
Interview Question Generator page.

Responsibilities:
    - Call core.interview_generator.generate_interview_questions()
      (LM Studio LLM) with the resume text.
    - Display the structured result in three tabs: Technical, Project,
      Behavioral.
    - Cache the result in session_state (interview.result); provide a
      "Regenerate" control, including a slider for how many questions
      per category to request.

Notes:
    - Mirrors the error-handling pattern used in suggestions_page.py:
      core.interview_generator.generate_interview_questions() encodes
      errors as a single message inside "technical" while leaving
      "project" and "behavioral" empty.
"""

from __future__ import annotations

from typing import Dict

import streamlit as st

from config import INTERVIEW_QUESTIONS_PER_CATEGORY
from core.interview_generator import generate_interview_questions
from utils.session_state import (
    require_resume,
    get_resume_text,
    get_interview_result,
    set_interview_result,
)
from utils.text_cleaning import format_list_as_bullets


def _is_error_result(result: Dict) -> str | None:
    """
    Check whether an interview-questions result represents an
    LLM/parsing error rather than real questions.

    Args:
        result: The interview questions result dict.

    Returns:
        The error message string if this result represents an error,
        or None if it looks like normal questions.
    """
    technical = result.get("technical", [])
    other_lists = result.get("project", []) + result.get("behavioral", [])

    if len(technical) == 1 and not other_lists:
        message = technical[0]
        if message.startswith("[LLM Error]") or "could not be parsed" in message.lower() or "No resume text" in message:
            return message

    return None


def _run_generation(num_questions: int) -> None:
    """
    Call generate_interview_questions() with the current resume text
    and store the result in session_state.

    Args:
        num_questions: Approximate number of questions to request per
                        category.
    """
    resume_text = get_resume_text()

    with st.spinner("Generating interview questions... this may take a minute."):
        result = generate_interview_questions(resume_text, questions_per_category=num_questions)

    set_interview_result(result)


def _render_questions(title: str, icon: str, questions: list) -> None:
    """
    Render a single category's questions as a numbered list.

    Args:
        title: Category title (e.g. "Technical").
        icon: Emoji shown next to the title.
        questions: List of question strings.
    """
    st.subheader(f"{icon} {title} Questions")

    if not questions:
        st.caption(f"No {title.lower()} questions were generated.")
        return

    for i, question in enumerate(questions, start=1):
        st.markdown(f"**{i}.** {question}")


def render() -> None:
    """
    Render the Interview Question Generator page.

    On first visit (no cached result), automatically runs generation
    with the default question count. A slider + "Regenerate" button
    allow re-running with a different count.
    """
    if not require_resume("Interview Prep"):
        return

    st.write(
        "Generate likely interview questions based on your resume — "
        "technical, project-specific, and behavioral."
    )

    num_questions = st.slider(
        "Questions per category",
        min_value=3,
        max_value=10,
        value=INTERVIEW_QUESTIONS_PER_CATEGORY,
        help="Approximate number of questions generated for each of the three categories.",
    )

    cached_result = get_interview_result()

    col1, _ = st.columns([1, 3])
    regenerate = col1.button(
        "🔄 Regenerate" if cached_result is not None else "✨ Generate Questions",
        type="primary",
    )

    if cached_result is None or regenerate:
        _run_generation(num_questions)

    result = get_interview_result()
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

    tab_technical, tab_project, tab_behavioral = st.tabs(
        ["🔧 Technical", "🚀 Projects", "🧑‍🤝‍🧑 Behavioral"]
    )

    with tab_technical:
        _render_questions("Technical", "🔧", result.get("technical", []))

    with tab_project:
        _render_questions("Project", "🚀", result.get("project", []))

    with tab_behavioral:
        _render_questions("Behavioral", "🧑‍🤝‍🧑", result.get("behavioral", []))

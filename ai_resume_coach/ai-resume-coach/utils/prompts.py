"""
utils/prompts.py
------------------
Centralized LLM prompt templates and shared prompt-building blocks.

Why this exists:
    - core/llm.py, core/suggestion_engine.py, core/interview_generator.py,
      and core/career_roadmap.py each build prompts inline today. This
      module exists as the single place to TUNE prompt wording without
      hunting through multiple files.
    - It does not change the public API of those modules — they remain
      self-contained and runnable on their own. This module re-exports
      the system messages and prompt builders they already use, plus a
      shared resume-formatting helper, so future edits can be made here
      and imported by those modules instead of duplicating strings.

Public API:
    - SYSTEM_MESSAGE_REVIEW
    - SYSTEM_MESSAGE_INTERVIEW
    - SYSTEM_MESSAGE_ROADMAP
    - SYSTEM_MESSAGE_CHAT
    - format_resume_block()       : Dict -> readable resume text block
    - build_review_prompt()
    - build_interview_prompt()
    - build_roadmap_prompt()
    - build_chat_prompt()

Usage (optional refactor):
    from utils.prompts import SYSTEM_MESSAGE_REVIEW, build_review_prompt
"""

from __future__ import annotations

from typing import Dict, List, Union


# ---------------------------------------------------------------------------
# Shared resume formatting
# ---------------------------------------------------------------------------
def format_resume_block(resume_data: Dict[str, Union[str, List[str]]]) -> str:
    """
    Convert a structured resume dictionary (flat pdf_parser schema:
    "name", "email", "phone", "skills", "education", "projects",
    "experience") into a readable plain-text block suitable for
    inclusion in an LLM prompt.

    This mirrors llm._resume_to_text() and
    text_cleaning.format_resume_for_llm() — kept here as the canonical
    version for new prompt-building code.

    Args:
        resume_data: The structured resume dictionary.

    Returns:
        A formatted multi-line string summarizing the resume.
    """
    def _section(title: str, items: Union[str, List[str]]) -> str:
        if isinstance(items, list):
            if not items:
                return f"{title}: (none listed)"
            bullet_items = "\n".join(f"  - {item}" for item in items)
            return f"{title}:\n{bullet_items}"
        return f"{title}: {items or '(not provided)'}"

    lines = [
        _section("Name", resume_data.get("name", "")),
        _section("Email", resume_data.get("email", "")),
        _section("Phone", resume_data.get("phone", "")),
        _section("Skills", resume_data.get("skills", [])),
        _section("Education", resume_data.get("education", [])),
        _section("Projects", resume_data.get("projects", [])),
        _section("Experience", resume_data.get("experience", [])),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resume Review (core/suggestion_engine.py)
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE_REVIEW: str = (
    "You are an expert resume reviewer and career coach. You give "
    "specific, actionable, and honest feedback. You do not invent "
    "facts about the candidate; you only work with the information "
    "provided."
)


def build_review_prompt(resume_text: str, job_description: str = "") -> str:
    """
    Build the user prompt for generate_resume_review() /
    review_resume().

    Args:
        resume_text: Resume text (raw or formatted via
                      format_resume_block()).
        job_description: Optional job description to tailor the
                          review against.

    Returns:
        A formatted prompt string.
    """
    jd_section = (
        f"\n\nTarget Job Description:\n{job_description.strip()}"
        if job_description.strip()
        else ""
    )

    return (
        "Review the following resume and provide improvement "
        "suggestions. Cover: (1) weak or vague bullet points and how "
        "to strengthen them, (2) any missing or underdeveloped "
        "sections, (3) formatting/structure issues for ATS "
        "compatibility, and (4) alignment with the target job "
        "description if one is provided. Present your feedback as a "
        "clear, organized list.\n\n"
        f"Resume:\n{resume_text.strip()}"
        f"{jd_section}"
    )


# ---------------------------------------------------------------------------
# Interview Questions (core/interview_generator.py)
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE_INTERVIEW: str = (
    "You are an experienced technical interviewer and career coach. "
    "You generate realistic, role-appropriate interview questions "
    "based on a candidate's background. You respond ONLY with valid "
    "JSON matching the exact schema requested. Do not include any "
    "explanation, commentary, or markdown formatting outside the JSON "
    "object."
)


def build_interview_prompt(resume_text: str, questions_per_category: int = 5) -> str:
    """
    Build the user prompt for generate_interview_questions().

    Args:
        resume_text: Resume text (raw or formatted via
                      format_resume_block()).
        questions_per_category: How many questions to request per
                                  category (technical/project/behavioral).

    Returns:
        A formatted prompt string instructing the LLM to return JSON
        with keys "technical", "project", "behavioral".
    """
    return (
        "Based on the resume below, generate interview questions and "
        "return them as a single JSON object with exactly these three "
        "keys:\n"
        '  - "technical": a list of strings — questions testing the '
        "candidate's knowledge of the specific skills, tools, and "
        "technologies listed in their resume.\n"
        '  - "project": a list of strings — deep-dive questions about '
        "the candidate's specific projects (their role, design "
        "decisions, challenges faced, and outcomes).\n"
        '  - "behavioral": a list of strings — soft-skill questions '
        "about teamwork, communication, conflict resolution, time "
        "management, and leadership, framed generally (not tied to a "
        "specific project).\n\n"
        f"Generate approximately {questions_per_category} questions per "
        "category. Respond with ONLY the JSON object — no extra text, "
        "no markdown code fences.\n\n"
        f"Resume:\n{resume_text.strip()}"
    )


# ---------------------------------------------------------------------------
# Career Roadmap (core/career_roadmap.py)
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE_ROADMAP: str = (
    "You are a career coach who creates realistic, actionable career "
    "development roadmaps based on a person's current resume. You "
    "respond ONLY with valid JSON matching the exact schema requested. "
    "Do not include any explanation, commentary, or markdown formatting "
    "outside the JSON object. Do not invent facts about the candidate "
    "beyond what is written in the resume."
)


def build_roadmap_prompt(resume_text: str, target_role: str = "") -> str:
    """
    Build the user prompt for generate_roadmap().

    Args:
        resume_text: Resume text (raw or formatted via
                      format_resume_block()).
        target_role: Optional target job title / career goal. If
                      empty, the model infers a sensible next step.

    Returns:
        A formatted prompt string instructing the LLM to return JSON
        with keys "skill_gaps", "recommended_technologies",
        "learning_roadmap", "suggested_projects".
    """
    role_description = (
        f'the target role of "{target_role.strip()}"'
        if target_role.strip()
        else (
            "a natural next step up from their current experience "
            "level (inferred from their resume)"
        )
    )

    return (
        f"Based on the resume below, create a career roadmap toward "
        f"{role_description}. Return your response as a single JSON "
        "object with exactly these four keys:\n"
        '  - "skill_gaps": a list of strings — specific skills, tools, '
        "or knowledge areas the candidate is currently missing or "
        "weak in for the target role.\n"
        '  - "recommended_technologies": a list of strings — specific '
        "technologies, frameworks, or tools the candidate should learn "
        "next, in rough priority order.\n"
        '  - "learning_roadmap": a list of JSON objects, each '
        "representing one stage of the plan, with exactly these keys:\n"
        '      - "stage": a short stage title (e.g. "Stage 1: '
        'Foundations").\n'
        '      - "duration": an estimated timeframe (e.g. "4-6 weeks").\n'
        '      - "topics": a list of strings — specific skills, '
        "concepts, or technologies to focus on during this stage.\n"
        '  - "suggested_projects": a list of strings — specific, '
        "hands-on project ideas (with enough detail to be actionable) "
        "that would build the skills identified above.\n\n"
        "Provide 3-6 items for each list, and 3-5 stages in "
        '"learning_roadmap". Respond with ONLY the JSON object — no '
        "extra text, no markdown code fences.\n\n"
        f"Resume:\n{resume_text.strip()}"
    )


# ---------------------------------------------------------------------------
# Resume Chatbot / RAG (core/llm.chat_with_resume)
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE_CHAT: str = (
    "You are a helpful assistant that answers questions about a "
    "specific person's resume. Use ONLY the provided resume context to "
    "answer. If the answer is not contained in the context, say that "
    "the resume does not provide that information rather than guessing."
)


def build_chat_prompt(question: str, context_chunks: List[str]) -> str:
    """
    Build the user prompt for chat_with_resume().

    Args:
        question: The user's question about their resume.
        context_chunks: Retrieved resume text chunks from
                         VectorStore.search().

    Returns:
        A formatted prompt string combining the retrieved context and
        the question.
    """
    context_text = (
        "\n---\n".join(chunk.strip() for chunk in context_chunks if chunk.strip())
        or "(No relevant resume content was found.)"
    )

    return f"Resume context:\n{context_text}\n\nQuestion: {question.strip()}"

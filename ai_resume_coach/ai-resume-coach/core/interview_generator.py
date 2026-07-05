"""
core/interview_generator.py
----------------------------
Module responsible for generating structured interview questions from
resume text using the local LM Studio LLM.

Generates three categories:
    - Technical questions  (skills/tools knowledge)
    - Project questions    (deep-dives into listed projects)
    - Behavioral questions (soft skills, teamwork, communication)

Depends on:
    - llm.py (_safe_generate, LLMClient)
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from llm import LLMClient, _safe_generate
from config import LLM_MAX_TOKENS_INTERVIEW


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
InterviewQuestions = Dict[str, List[str]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUESTION_SCHEMA_KEYS: List[str] = ["technical", "project", "behavioral"]

DEFAULT_QUESTIONS_PER_CATEGORY: int = 5

SYSTEM_MESSAGE: str = (
    "You are an experienced technical interviewer and career coach. "
    "You generate realistic, role-appropriate interview questions based "
    "on a candidate's resume. You respond ONLY with valid JSON matching "
    "the exact schema requested. Do not include any explanation, "
    "commentary, or markdown formatting outside the JSON object."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_interview_prompt(
    resume_text: str,
    questions_per_category: int = DEFAULT_QUESTIONS_PER_CATEGORY,
) -> str:
    """Build the user prompt for interview question generation."""
    return (
        "Based on the resume below, generate interview questions and "
        "return them as a single JSON object with exactly these three keys:\n"
        '  - "technical": a list of strings — questions testing the '
        "candidate's knowledge of the specific skills, tools, and "
        "technologies listed in their resume.\n"
        '  - "project": a list of strings — deep-dive questions about '
        "the candidate's specific projects (role, design decisions, "
        "challenges, outcomes).\n"
        '  - "behavioral": a list of strings — soft-skill questions about '
        "teamwork, communication, conflict resolution, and time management.\n\n"
        f"Generate approximately {questions_per_category} questions per "
        "category. Respond with ONLY the JSON object — no extra text, "
        "no markdown code fences.\n\n"
        f"Resume:\n{resume_text.strip()}"
    )


# ---------------------------------------------------------------------------
# Response parsing / normalization
# ---------------------------------------------------------------------------
def _extract_json_block(raw_response: str) -> Optional[str]:
    """Extract JSON object from raw LLM response, handling code fences."""
    if not raw_response:
        return None

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```", raw_response, re.DOTALL
    )
    if fenced_match:
        return fenced_match.group(1)

    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_response[start : end + 1]

    return None


def _normalize_questions(parsed: Dict) -> InterviewQuestions:
    """Normalize a parsed JSON dict into the fixed InterviewQuestions schema."""
    normalized: InterviewQuestions = {}

    for key in QUESTION_SCHEMA_KEYS:
        value = parsed.get(key, [])
        if isinstance(value, list):
            normalized[key] = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            normalized[key] = [value.strip()] if value.strip() else []
        else:
            normalized[key] = []

    return normalized


def _empty_questions(error_message: Optional[str] = None) -> InterviewQuestions:
    """Return an empty questions result, optionally with an error message."""
    result: InterviewQuestions = {key: [] for key in QUESTION_SCHEMA_KEYS}
    if error_message:
        result["technical"] = [error_message]
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_interview_questions(
    resume_text: str,
    questions_per_category: int = DEFAULT_QUESTIONS_PER_CATEGORY,
    client: Optional[LLMClient] = None,
) -> InterviewQuestions:
    """
    Generate technical, project, and behavioral interview questions
    based on a candidate's resume using the local LLM.

    Args:
        resume_text: The raw or cleaned text of the candidate's resume.
        questions_per_category: Number of questions per category.
        client: Optional LLMClient instance. Uses the default if None.

    Returns:
        A dictionary with keys: "technical", "project", "behavioral".
        Never raises — errors are encoded in the result dict.
    """
    if not resume_text or not resume_text.strip():
        return _empty_questions(
            "No resume text was provided to generate questions from."
        )

    prompt = build_interview_prompt(resume_text, questions_per_category)

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    raw_response = _safe_generate(
        client, messages, max_tokens=LLM_MAX_TOKENS_INTERVIEW
    )

    if raw_response.startswith("[LLM Error]"):
        return _empty_questions(raw_response)

    json_block = _extract_json_block(raw_response)
    if json_block is None:
        return _empty_questions(
            "The AI's response could not be parsed as JSON. Try again, "
            "or check that LM Studio's model is responding correctly."
        )

    try:
        parsed = json.loads(json_block)
    except json.JSONDecodeError:
        return _empty_questions(
            "The AI returned malformed JSON and the response could not "
            "be parsed. Try again."
        )

    if not isinstance(parsed, dict):
        return _empty_questions(
            "The AI's response was valid JSON but not in the expected "
            "object format."
        )

    return _normalize_questions(parsed)


# ---------------------------------------------------------------------------
# Manual test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_resume_text = (
        "Jane Doe\njane@example.com | +1 555 123 4567\n\n"
        "Skills: Python, Machine Learning, FAISS, Streamlit, SQL\n\n"
        "Experience:\nData Science Intern, Acme Corp\n"
        "Worked on ML pipelines and NLP models.\n\n"
        "Projects:\nBuilt a resume analyzer using NLP and vector search.\n\n"
        "Education:\nB.Tech in Computer Science, 2024"
    )

    result = generate_interview_questions(sample_resume_text)
    print(json.dumps(result, indent=2))
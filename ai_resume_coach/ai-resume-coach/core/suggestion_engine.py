"""
core/suggestion_engine.py
--------------------------
Module responsible for generating a structured resume review using the
local LM Studio LLM.

Given raw resume text, this module prompts the LLM to analyze the
resume and return a structured assessment covering:
    - Strengths
    - Weaknesses
    - Missing sections
    - Improvement suggestions

The result is always returned as a Python dictionary with a fixed
schema, regardless of whether the LLM responds with valid JSON,
malformed JSON, or fails entirely.

Depends on:
    - llm.py (LLMClient, _safe_generate, LLMConnectionError)
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from llm import LLMClient, _safe_generate
from config import LLM_MAX_TOKENS_REVIEW


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ReviewResult = Dict[str, List[str]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REVIEW_SCHEMA_KEYS: List[str] = [
    "strengths",
    "weaknesses",
    "missing_sections",
    "improvement_suggestions",
]

SYSTEM_MESSAGE: str = (
    "You are an expert resume reviewer and career coach. You analyze "
    "resumes and respond ONLY with valid JSON matching the exact schema "
    "requested. Do not include any explanation, commentary, or markdown "
    "formatting outside the JSON object. Do not invent facts about the "
    "candidate beyond what is written in the resume."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_review_prompt(resume_text: str, job_description: str = "") -> str:
    """Build the user prompt for the resume review."""
    jd_section = (
        f"\n\nTarget Job Description:\n{job_description.strip()}"
        if job_description.strip()
        else ""
    )

    return (
        "Analyze the following resume and return your assessment as a "
        "single JSON object with exactly these four keys:\n"
        '  - "strengths": a list of strings describing what the resume does well.\n'
        '  - "weaknesses": a list of strings describing problems with '
        "content, wording, or presentation.\n"
        '  - "missing_sections": a list of strings naming standard resume '
        "sections that are absent or underdeveloped.\n"
        '  - "improvement_suggestions": a list of strings with specific, '
        "actionable recommendations to improve the resume.\n\n"
        "Each list should contain 3-6 concise items. Respond with ONLY "
        "the JSON object — no extra text, no markdown code fences.\n\n"
        f"Resume:\n{resume_text.strip()}"
        f"{jd_section}"
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


def _normalize_review(parsed: Dict) -> ReviewResult:
    """Normalize a parsed JSON dict into the fixed ReviewResult schema."""
    normalized: ReviewResult = {}

    for key in REVIEW_SCHEMA_KEYS:
        value = parsed.get(key, [])
        if isinstance(value, list):
            normalized[key] = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            normalized[key] = [value.strip()] if value.strip() else []
        else:
            normalized[key] = []

    return normalized


def _empty_review(error_message: Optional[str] = None) -> ReviewResult:
    """Return an empty review result, optionally with an error message."""
    result: ReviewResult = {key: [] for key in REVIEW_SCHEMA_KEYS}
    if error_message:
        result["weaknesses"] = [error_message]
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def review_resume(
    resume_text: str,
    job_description: str = "",
    client: Optional[LLMClient] = None,
) -> ReviewResult:
    """
    Generate a structured review of a resume using the local LLM.

    Args:
        resume_text: The raw or cleaned text of the resume to review.
        job_description: Optional job description to tailor feedback.
        client: Optional LLMClient instance. Uses the default if None.

    Returns:
        A dictionary with keys: "strengths", "weaknesses",
        "missing_sections", "improvement_suggestions".
        Never raises — errors are encoded in the result dict.
    """
    if not resume_text or not resume_text.strip():
        return _empty_review("No resume text was provided to review.")

    prompt = build_review_prompt(resume_text, job_description)

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    # Use _safe_generate which handles the system-role merge, error
    # catching, and returns "[LLM Error] ..." on failure.
    raw_response = _safe_generate(client, messages, max_tokens=LLM_MAX_TOKENS_REVIEW)

    if raw_response.startswith("[LLM Error]"):
        return _empty_review(raw_response)

    json_block = _extract_json_block(raw_response)
    if json_block is None:
        return _empty_review(
            "The AI's response could not be parsed as JSON. Try again, "
            "or check that LM Studio's model is responding correctly."
        )

    try:
        parsed = json.loads(json_block)
    except json.JSONDecodeError:
        return _empty_review(
            "The AI returned malformed JSON and the response could not "
            "be parsed. Try again."
        )

    if not isinstance(parsed, dict):
        return _empty_review(
            "The AI's response was valid JSON but not in the expected "
            "object format."
        )

    return _normalize_review(parsed)


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

    result = review_resume(sample_resume_text)
    print(json.dumps(result, indent=2))
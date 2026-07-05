"""
core/career_roadmap.py
-----------------------
Module responsible for generating a structured career roadmap from
resume text and an optional target role using the local LM Studio LLM.

Generates:
    - Skill gaps
    - Recommended technologies
    - Learning roadmap (staged plan)
    - Suggested projects

Depends on:
    - llm.py (_safe_generate, LLMClient)
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Union

from llm import LLMClient, _safe_generate
from config import LLM_MAX_TOKENS_ROADMAP


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RoadmapStage = Dict[str, Union[str, List[str]]]
RoadmapResult = Dict[str, Union[List[str], List[RoadmapStage]]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROADMAP_SCHEMA_KEYS: List[str] = [
    "skill_gaps",
    "recommended_technologies",
    "learning_roadmap",
    "suggested_projects",
]

ROADMAP_STAGE_KEYS: List[str] = ["stage", "duration", "topics"]

DEFAULT_TARGET_ROLE_DESCRIPTION: str = (
    "a natural next step up from their current experience level "
    "(inferred from their resume)"
)

SYSTEM_MESSAGE: str = (
    "You are a career coach who creates realistic, actionable career "
    "development roadmaps based on a person's current resume. You "
    "respond ONLY with valid JSON matching the exact schema requested. "
    "Do not include any explanation, commentary, or markdown formatting "
    "outside the JSON object. Do not invent facts about the candidate "
    "beyond what is written in the resume."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_roadmap_prompt(resume_text: str, target_role: str = "") -> str:
    """Build the user prompt for career roadmap generation."""
    role_description = (
        f'the target role of "{target_role.strip()}"'
        if target_role.strip()
        else DEFAULT_TARGET_ROLE_DESCRIPTION
    )

    return (
        f"Based on the resume below, create a career roadmap toward "
        f"{role_description}. Return your response as a single JSON "
        "object with exactly these four keys:\n"
        '  - "skill_gaps": a list of strings — skills the candidate is '
        "currently missing for the target role.\n"
        '  - "recommended_technologies": a list of strings — tools and '
        "frameworks to learn next, in priority order.\n"
        '  - "learning_roadmap": a list of JSON objects, each with:\n'
        '      - "stage": short stage title (e.g. "Stage 1: Foundations")\n'
        '      - "duration": estimated timeframe (e.g. "4-6 weeks")\n'
        '      - "topics": a list of strings — topics to cover in this stage\n'
        '  - "suggested_projects": a list of strings — specific hands-on '
        "project ideas to build the identified skills.\n\n"
        "Provide 3-6 items per list, and 3-5 stages in learning_roadmap. "
        "Respond with ONLY the JSON object — no extra text, no markdown "
        "code fences.\n\n"
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


def _normalize_string_list(value: object) -> List[str]:
    """Normalize a value into a list of non-empty stripped strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _normalize_roadmap_stages(value: object) -> List[RoadmapStage]:
    """Normalize the learning_roadmap value into a list of stage dicts."""
    if not isinstance(value, list):
        return []

    normalized_stages: List[RoadmapStage] = []

    for item in value:
        if isinstance(item, dict):
            stage_title = str(item.get("stage", "")).strip()
            duration = str(item.get("duration", "")).strip()
            topics = _normalize_string_list(item.get("topics", []))

            if not stage_title and not duration and not topics:
                continue

            normalized_stages.append({
                "stage": stage_title,
                "duration": duration,
                "topics": topics,
            })
        elif isinstance(item, str):
            stripped = item.strip()
            if stripped:
                normalized_stages.append(
                    {"stage": stripped, "duration": "", "topics": []}
                )

    return normalized_stages


def _normalize_roadmap(parsed: Dict) -> RoadmapResult:
    """Normalize a parsed JSON dict into the fixed RoadmapResult schema."""
    return {
        "skill_gaps": _normalize_string_list(parsed.get("skill_gaps", [])),
        "recommended_technologies": _normalize_string_list(
            parsed.get("recommended_technologies", [])
        ),
        "learning_roadmap": _normalize_roadmap_stages(
            parsed.get("learning_roadmap", [])
        ),
        "suggested_projects": _normalize_string_list(
            parsed.get("suggested_projects", [])
        ),
    }


def _empty_roadmap(error_message: Optional[str] = None) -> RoadmapResult:
    """Return an empty roadmap result, optionally with an error message."""
    result: RoadmapResult = {
        "skill_gaps": [],
        "recommended_technologies": [],
        "learning_roadmap": [],
        "suggested_projects": [],
    }
    if error_message:
        result["skill_gaps"] = [error_message]
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_roadmap(
    resume_text: str,
    target_role: str = "",
    client: Optional[LLMClient] = None,
) -> RoadmapResult:
    """
    Generate a structured career roadmap based on a candidate's resume
    and an optional target role, using the local LLM.

    Args:
        resume_text: The raw or cleaned text of the candidate's resume.
        target_role: Optional target job title. If empty, the model
                      infers a sensible next-step role from the resume.
        client: Optional LLMClient instance. Uses the default if None.

    Returns:
        A dictionary with keys: "skill_gaps", "recommended_technologies",
        "learning_roadmap", "suggested_projects".
        Never raises — errors are encoded in the result dict.
    """
    if not resume_text or not resume_text.strip():
        return _empty_roadmap(
            "No resume text was provided to generate a roadmap from."
        )

    prompt = build_roadmap_prompt(resume_text, target_role)

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    raw_response = _safe_generate(
        client, messages, max_tokens=LLM_MAX_TOKENS_ROADMAP
    )

    if raw_response.startswith("[LLM Error]"):
        return _empty_roadmap(raw_response)

    json_block = _extract_json_block(raw_response)
    if json_block is None:
        return _empty_roadmap(
            "The AI's response could not be parsed as JSON. Try again, "
            "or check that LM Studio's model is responding correctly."
        )

    try:
        parsed = json.loads(json_block)
    except json.JSONDecodeError:
        return _empty_roadmap(
            "The AI returned malformed JSON and the response could not "
            "be parsed. Try again."
        )

    if not isinstance(parsed, dict):
        return _empty_roadmap(
            "The AI's response was valid JSON but not in the expected "
            "object format."
        )

    return _normalize_roadmap(parsed)


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

    result = generate_roadmap(
        sample_resume_text, target_role="Machine Learning Engineer"
    )
    print(json.dumps(result, indent=2))
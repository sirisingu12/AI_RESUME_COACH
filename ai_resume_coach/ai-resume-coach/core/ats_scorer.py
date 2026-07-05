"""
core/ats_scorer.py
-------------------
Module responsible for computing an ATS (Applicant Tracking System) style
compatibility score between a parsed resume and a job description (JD).

Scoring breakdown (overall score out of 100):
    - Skills Match     : 40 points
    - Experience Match : 25 points
    - Projects Match   : 15 points
    - Education Match  : 10 points
    - Resume Structure : 10 points

Scoring logic:
    1. A job description MUST be provided for a meaningful score.
       If no JD is given, only the structure score (10 pts) is computed
       and the remaining 90 pts are left at 0. This prevents the bug
       where every resume scores 100/100 with no JD.

    2. For each resume section (skills, experience, projects, education),
       we extract keywords from both the section and the JD, then score
       based on what fraction of the JD keywords appear in that section.

    3. The structure score checks whether all standard sections are
       present AND non-empty. An empty experience list is penalised.

    4. Section scores are capped by a per-section keyword density cap
       (MAX_SECTION_COVERAGE) so a resume cannot hit 40/40 on skills
       just by having a very short JD with one word the resume contains.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Union

from config import ATS_WEIGHTS as WEIGHTS


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ParsedResume = Dict[str, Union[str, List[str]]]
ATSResult = Dict[str, Union[int, float, List[str]]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on",
    "for", "with", "at", "by", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "we", "you", "your", "our", "their", "they", "he", "she", "will",
    "shall", "can", "may", "should", "would", "must", "have", "has",
    "had", "do", "does", "did", "not", "no", "so", "such", "than",
    "into", "about", "etc", "using", "use", "experience", "years",
    "year", "work", "working", "ability", "skills", "skill", "strong",
    "knowledge", "including", "responsible", "responsibilities", "role",
}

MIN_KEYWORD_LENGTH: int = 2

# Maximum coverage ratio a single section can score against the JD.
# Even if a section contains ALL JD keywords, it is capped at this
# fraction of max_points — preventing trivially short JDs from giving
# inflated scores. 0.85 = a section can score at most 85% of its
# max points purely from keyword overlap.
MAX_SECTION_COVERAGE: float = 0.85

# Standard resume sections used for structure scoring.
EXPECTED_STRUCTURE_FIELDS: List[str] = [
    "name",
    "email",
    "phone",
    "skills",
    "education",
    "experience",
    "projects",
]


# ---------------------------------------------------------------------------
# 1. KEYWORD EXTRACTION
# ---------------------------------------------------------------------------
def extract_keywords(text: str) -> Set[str]:
    """
    Convert free-form text into a normalized set of keyword tokens.

    Args:
        text: Any raw text (resume section or JD).

    Returns:
        A set of unique, lowercased keyword tokens with stopwords removed.
    """
    if not text:
        return set()

    lowered = text.lower()
    raw_tokens = re.findall(r"[a-z0-9][a-z0-9+#.]*", lowered)

    keywords: Set[str] = set()
    for token in raw_tokens:
        cleaned = token.strip(".")
        if len(cleaned) < MIN_KEYWORD_LENGTH:
            continue
        if cleaned in STOPWORDS:
            continue
        keywords.add(cleaned)

    return keywords


def _flatten_section(section_value: Union[str, List[str]]) -> str:
    """Flatten a resume section (str or list) into a single text blob."""
    if isinstance(section_value, list):
        return " ".join(str(item) for item in section_value)
    return str(section_value or "")


# ---------------------------------------------------------------------------
# 2. SECTION-LEVEL MATCH SCORING
# ---------------------------------------------------------------------------
def calculate_section_match_score(
    section_text: str,
    jd_keywords: Set[str],
    max_points: int,
    section_name: str = "",
) -> float:
    """
    Score how well a single resume section covers the JD keywords.

    Key fix: If jd_keywords is empty (no JD provided), returns 0.0
    instead of max_points. This prevents every resume from scoring
    100/100 when no JD is entered.

    Also applies a MAX_SECTION_COVERAGE cap so very short JDs don't
    give unrealistically perfect scores.

    Args:
        section_text: The combined text of the resume section.
        jd_keywords: Keywords extracted from the job description.
        max_points: Maximum points this section can contribute.
        section_name: Optional label for debugging.

    Returns:
        A float score in [0, max_points * MAX_SECTION_COVERAGE].
    """
    # FIX: No JD = 0 points, not full marks.
    if not jd_keywords:
        return 0.0

    section_keywords = extract_keywords(section_text)
    if not section_keywords:
        return 0.0

    matched = section_keywords & jd_keywords
    if not matched:
        return 0.0

    # Score = fraction of JD keywords found in this section.
    coverage_ratio = len(matched) / len(jd_keywords)

    # Apply cap: even a perfect match only earns MAX_SECTION_COVERAGE
    # of the available points, leaving room for the JD to differentiate.
    coverage_ratio = min(coverage_ratio, MAX_SECTION_COVERAGE)

    return round(coverage_ratio * max_points, 2)


# ---------------------------------------------------------------------------
# 3. RESUME STRUCTURE SCORING (JD-independent)
# ---------------------------------------------------------------------------
def calculate_structure_score(
    parsed_resume: ParsedResume,
    max_points: int = WEIGHTS["structure"],
) -> float:
    """
    Score the completeness of the resume's structure.

    Each of the EXPECTED_STRUCTURE_FIELDS is worth an equal share of
    max_points. A field is "present" only if it is a non-empty string
    (for name/email/phone) or a non-empty list (for skills/etc.).

    An empty experience list correctly scores 0 for that field —
    it is not ignored or given a pass.

    Args:
        parsed_resume: The structured resume dictionary.
        max_points: Maximum points the structure score can contribute.

    Returns:
        A float score in [0, max_points].
    """
    total_fields = len(EXPECTED_STRUCTURE_FIELDS)
    if total_fields == 0:
        return 0.0

    points_per_field = max_points / total_fields
    score = 0.0

    for field in EXPECTED_STRUCTURE_FIELDS:
        value = parsed_resume.get(field)
        if value:  # Non-empty string or non-empty list.
            score += points_per_field

    return round(score, 2)


# ---------------------------------------------------------------------------
# 4. MISSING KEYWORDS
# ---------------------------------------------------------------------------
def find_missing_keywords(
    parsed_resume: ParsedResume,
    jd_keywords: Set[str],
) -> List[str]:
    """
    Identify JD keywords not found anywhere in the resume.

    Searches across skills, education, projects, and experience combined.

    Args:
        parsed_resume: The structured resume dictionary.
        jd_keywords: Keywords extracted from the job description.

    Returns:
        A sorted list of JD keywords missing from the resume.
    """
    combined_resume_text = " ".join(
        _flatten_section(parsed_resume.get(field, []))
        for field in ("skills", "education", "projects", "experience")
    )

    resume_keywords = extract_keywords(combined_resume_text)
    missing = jd_keywords - resume_keywords

    return sorted(missing)


# ---------------------------------------------------------------------------
# 5. MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def calculate_ats_score(
    parsed_resume: ParsedResume,
    job_description: str,
) -> ATSResult:
    """
    Compute the full ATS score breakdown for a resume against a JD.

    If no job description is provided:
        - All JD-dependent scores (skills, experience, projects,
          education) are 0 — NOT full marks.
        - Only the structure score is computed (max 10 points).
        - "no_jd" flag is set to True so the UI can show a message
          prompting the user to add a JD for a real score.

    Args:
        parsed_resume: The structured resume dictionary from parser.py.
        job_description: The raw text of the target job description.

    Returns:
        A dictionary with keys:
            - "overall_score": int, 0-100
            - "skills_score": float, 0-40
            - "experience_score": float, 0-25
            - "projects_score": float, 0-15
            - "education_score": float, 0-10
            - "structure_score": float, 0-10
            - "missing_keywords": List[str]
            - "no_jd": bool — True if no JD was provided
    """
    jd_keywords = extract_keywords(job_description)
    no_jd = len(jd_keywords) == 0

    skills_text = _flatten_section(parsed_resume.get("skills", []))
    experience_text = _flatten_section(parsed_resume.get("experience", []))
    projects_text = _flatten_section(parsed_resume.get("projects", []))
    education_text = _flatten_section(parsed_resume.get("education", []))

    skills_score = calculate_section_match_score(
        skills_text, jd_keywords, WEIGHTS["skills"], "skills"
    )
    experience_score = calculate_section_match_score(
        experience_text, jd_keywords, WEIGHTS["experience"], "experience"
    )
    projects_score = calculate_section_match_score(
        projects_text, jd_keywords, WEIGHTS["projects"], "projects"
    )
    education_score = calculate_section_match_score(
        education_text, jd_keywords, WEIGHTS["education"], "education"
    )
    structure_score = calculate_structure_score(
        parsed_resume, WEIGHTS["structure"]
    )

    overall_score = round(
        skills_score
        + experience_score
        + projects_score
        + education_score
        + structure_score
    )

    missing_keywords = (
        find_missing_keywords(parsed_resume, jd_keywords)
        if jd_keywords
        else []
    )

    return {
        "overall_score": overall_score,
        "skills_score": skills_score,
        "experience_score": experience_score,
        "projects_score": projects_score,
        "education_score": education_score,
        "structure_score": structure_score,
        "missing_keywords": missing_keywords,
        "no_jd": no_jd,
    }


# ---------------------------------------------------------------------------
# Manual test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    anvith_resume: ParsedResume = {
        "name": "Anvith Sai",
        "email": "gullapellyanvithsai@gmail.com",
        "phone": "+91 70137 04284",
        "skills": ["Python", "C", "JavaScript", "Pandas", "NumPy",
                   "Matplotlib", "Seaborn", "HTML", "CSS", "Node.js",
                   "React Native", "MySQL", "MongoDB", "VS Code",
                   "Google Colab", "Power BI"],
        "education": ["B.Tech in Computer Science Engineering (AI & ML), SR University, 2028"],
        "experience": [],
        "projects": [
            "Brain Tumor Detection Python TensorFlow CNN OpenCV",
            "AI Government Service Finder Python NLP",
            "Car Price Prediction Python Scikit-learn Machine Learning",
            "Data Visualization Dashboard Python Pandas Matplotlib Power BI",
            "Vehicle Service Booking System HTML CSS JavaScript",
        ],
    }

    # Test 1: No JD — should NOT be 100
    print("=== No JD ===")
    result = calculate_ats_score(anvith_resume, "")
    print(json.dumps(result, indent=2))

    # Test 2: Relevant JD
    print("\n=== With JD ===")
    jd = (
        "Looking for a Python developer with machine learning, deep learning, "
        "TensorFlow, NLP, and data visualization experience. "
        "Knowledge of React and MongoDB is a plus."
    )
    result = calculate_ats_score(anvith_resume, jd)
    print(json.dumps(result, indent=2))
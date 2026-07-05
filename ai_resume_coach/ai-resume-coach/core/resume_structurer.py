"""
core/resume_structurer.py
--------------------------
Module responsible for converting raw resume text (e.g. as extracted
by core/pdf_parser.py) into structured information.

Extracts:
    - Contact information (name, email, phone)
    - Skills
    - Education
    - Experience
    - Projects
    - Certifications

Output:
    A dictionary with a fixed schema (see StructuredResume), suitable
    for direct use by downstream consumers:
        - ats_scorer.py
        - keyword_gap.py
        - jd_matcher.py
        - career_roadmap.py

Design notes:
    - Parsing is entirely regex/heuristic-based; no LLM or network
      calls are made here, keeping this module fast and deterministic.
    - Missing sections never raise — they are returned as empty
      lists/strings so downstream consumers can rely on the schema
      always being present.
    - Logging is used throughout (at DEBUG/INFO/WARNING level) to aid
      debugging when a resume's sections are not detected as expected,
      without printing to stdout directly.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Avoid duplicate handlers if this module is imported multiple times
    # or the root logger is configured elsewhere (e.g. by Streamlit).
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ContactInfo = Dict[str, str]
StructuredResume = Dict[str, Union[ContactInfo, List[str]]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Section header keywords used to split the resume into blocks. Keys
# are the canonical section names; values are alternative header
# phrasings commonly found in resumes (matched case-insensitively,
# with punctuation stripped).
SECTION_HEADERS: Dict[str, List[str]] = {
    "skills": [
        "skills",
        "technical skills",
        "core competencies",
        "key skills",
        "skills and abilities",
        "technical proficiencies",
    ],
    "education": [
        "education",
        "academic background",
        "academic qualifications",
        "educational qualifications",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "internships",
        "internship experience",
        "work history",
    ],
    "projects": [
        "projects",
        "academic projects",
        "personal projects",
        "key projects",
        "project experience",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "licenses and certifications",
        "certifications and courses",
        "courses and certifications",
    ],
}

# Maximum number of words a line can have to plausibly be a section
# header (longer lines are body text, not headers).
MAX_HEADER_WORDS: int = 5

# Maximum number of leading lines to scan when looking for the
# candidate's name.
NAME_SEARCH_LINE_LIMIT: int = 5

# Bullet/marker characters stripped from the start of list entries.
BULLET_CHARS: str = " -:\t*•▪·●○"


# ---------------------------------------------------------------------------
# 1. TEXT CLEANING
# ---------------------------------------------------------------------------
def clean_text(raw_text: str) -> str:
    """
    Normalize raw resume text for easier downstream parsing.

    - Collapses multiple spaces/tabs into a single space.
    - Strips trailing whitespace from each line.
    - Removes blank lines.
    - Strips leading/trailing whitespace from the whole text.

    Args:
        raw_text: Text as extracted from the resume (e.g. via pypdf).

    Returns:
        Cleaned text, ready for line-based and regex-based processing.
        Returns "" if `raw_text` is empty or whitespace-only.
    """
    if not raw_text or not raw_text.strip():
        logger.debug("clean_text received empty input.")
        return ""

    text = re.sub(r"[ \t]+", " ", raw_text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    cleaned = "\n".join(lines).strip()
    logger.debug("clean_text produced %d non-empty lines.", len(lines))
    return cleaned


# ---------------------------------------------------------------------------
# 2. CONTACT INFORMATION EXTRACTION
# ---------------------------------------------------------------------------
def extract_email(text: str) -> str:
    """
    Extract the first email address found in the text.

    Args:
        text: Cleaned resume text.

    Returns:
        The email address as a string, or "" if none is found.
    """
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if match:
        return match.group(0)

    logger.debug("No email address found in resume text.")
    return ""


def extract_phone(text: str) -> str:
    """
    Extract the first phone-number-like sequence from the text.

    Supports common formats: optional country code, separators of
    space/dot/dash/parentheses, and at least 10 digits total.

    Args:
        text: Cleaned resume text.

    Returns:
        The phone number as a string, or "" if none is found.
    """
    pattern = r"(\+?\d{1,3}[\s-]?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}"

    for line in text.splitlines():
        match = re.search(pattern, line)
        if match:
            candidate = match.group(0).strip()
            digit_count = sum(ch.isdigit() for ch in candidate)
            # Require at least 10 digits to avoid matching dates,
            # years, or partial numbers.
            if digit_count >= 10:
                return candidate

    logger.debug("No phone number found in resume text.")
    return ""


def extract_name(text: str, header_keywords: Optional[set] = None) -> str:
    """
    Heuristically extract the candidate's name.

    Assumption: the name is typically one of the first few non-empty
    lines of the resume that does not look like an email, phone
    number, section header, or an unusually long line of body text.

    Args:
        text: Cleaned resume text.
        header_keywords: Optional precomputed set of all section
                          header keywords (lowercase) to skip. If not
                          provided, derived from SECTION_HEADERS.

    Returns:
        The best-guess candidate name, or "" if none can be determined.
    """
    if header_keywords is None:
        header_keywords = {
            kw for kws in SECTION_HEADERS.values() for kw in kws
        }

    lines = text.splitlines()

    for line in lines[:NAME_SEARCH_LINE_LIMIT]:
        candidate = line.strip()
        lower_candidate = candidate.lower()

        if not candidate:
            continue
        if "@" in candidate:
            continue
        if any(ch.isdigit() for ch in candidate):
            continue
        if lower_candidate in header_keywords:
            continue
        if len(candidate.split()) > 6:
            continue

        return candidate

    logger.warning("Could not confidently determine candidate name.")
    return ""


def extract_contact_info(text: str) -> ContactInfo:
    """
    Extract all contact information (name, email, phone) from the
    resume text.

    Args:
        text: Cleaned resume text.

    Returns:
        A dictionary with keys "name", "email", and "phone", each a
        string (possibly empty if not found).
    """
    header_keywords = {kw for kws in SECTION_HEADERS.values() for kw in kws}

    return {
        "name": extract_name(text, header_keywords),
        "email": extract_email(text),
        "phone": extract_phone(text),
    }


# ---------------------------------------------------------------------------
# 3. SECTION SPLITTING
# ---------------------------------------------------------------------------
def _normalize_header_line(line: str) -> str:
    """
    Normalize a line for header matching: lowercase, strip digits and
    punctuation, collapse whitespace.

    Args:
        line: A raw line of resume text.

    Returns:
        The normalized line, suitable for comparison against
        SECTION_HEADERS values.
    """
    normalized = re.sub(r"[^a-z\s]", "", line.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _match_section_header(normalized_line: str) -> Optional[str]:
    """
    Check whether a normalized line corresponds to a known section
    header.

    Args:
        normalized_line: A line of text, normalized via
                          _normalize_header_line().

    Returns:
        The canonical section name if the line matches a known header
        and is short enough to plausibly BE a header (not body text),
        otherwise None.
    """
    if not normalized_line:
        return None
    if len(normalized_line.split()) > MAX_HEADER_WORDS:
        return None

    for section_name, keywords in SECTION_HEADERS.items():
        if normalized_line in keywords:
            return section_name

    return None


def split_into_sections(text: str) -> Dict[str, List[str]]:
    """
    Split resume text into named sections based on header keywords.

    Scans each line; when a line matches a known section header, all
    subsequent lines are assigned to that section until the next
    recognized header is found. Lines before the first recognized
    header are ignored (typically the contact info block).

    Args:
        text: Cleaned resume text.

    Returns:
        A dictionary mapping each canonical section name ("skills",
        "education", "experience", "projects", "certifications") to a
        list of raw lines belonging to that section. Sections not
        found in the resume map to an empty list.
    """
    sections: Dict[str, List[str]] = {key: [] for key in SECTION_HEADERS}
    current_section: Optional[str] = None

    for line in text.splitlines():
        normalized = _normalize_header_line(line)
        matched_section = _match_section_header(normalized)

        if matched_section:
            logger.debug("Detected section header: '%s' -> %s", line, matched_section)
            current_section = matched_section
            continue  # Don't include the header line itself.

        if current_section:
            sections[current_section].append(line.strip())

    for section_name, lines in sections.items():
        if not lines:
            logger.info("Section '%s' was not found in the resume.", section_name)

    return sections


# ---------------------------------------------------------------------------
# 4. FIELD EXTRACTION FROM SECTIONS
# ---------------------------------------------------------------------------
def _strip_bullets(lines: List[str]) -> List[str]:
    """
    Strip leading bullet/marker characters and whitespace from each
    line, dropping any lines that become empty.

    Args:
        lines: Raw lines from a resume section.

    Returns:
        A list of cleaned, non-empty strings.
    """
    return [line.strip(BULLET_CHARS) for line in lines if line.strip(BULLET_CHARS)]


def extract_skills(skill_lines: List[str]) -> List[str]:
    """
    Convert raw lines from the "skills" section into a flat,
    deduplicated list of individual skills.

    Handles skills listed comma-separated, separated by
    pipes/slashes/bullets, or one per line.

    Args:
        skill_lines: Raw lines belonging to the skills section.

    Returns:
        A deduplicated list of skill strings, trimmed of whitespace
        and surrounding punctuation.
    """
    skills: List[str] = []

    for line in skill_lines:
        parts = re.split(r"[,|/•▪·●○]", line)
        for part in parts:
            cleaned = part.strip(BULLET_CHARS)
            if cleaned:
                skills.append(cleaned)

    seen: set = set()
    unique_skills: List[str] = []
    for skill in skills:
        key = skill.lower()
        if key not in seen:
            seen.add(key)
            unique_skills.append(skill)

    return unique_skills


def extract_education(education_lines: List[str]) -> List[str]:
    """
    Convert raw lines from the "education" section into a list of
    individual education entries.

    Each non-empty, bullet-stripped line is treated as a separate
    entry (degree, institution, year, etc. are kept together as
    written).

    Args:
        education_lines: Raw lines belonging to the education section.

    Returns:
        A list of cleaned education entry strings.
    """
    return _strip_bullets(education_lines)


def extract_experience(experience_lines: List[str]) -> List[str]:
    """
    Convert raw lines from the "experience" section into a list of
    individual experience entries.

    Each non-empty, bullet-stripped line becomes one entry (e.g. job
    title, company, dates, or a responsibility bullet point).

    Args:
        experience_lines: Raw lines belonging to the experience
                           section.

    Returns:
        A list of cleaned experience entry strings.
    """
    return _strip_bullets(experience_lines)


def extract_projects(project_lines: List[str]) -> List[str]:
    """
    Convert raw lines from the "projects" section into a list of
    individual project entries.

    Each non-empty, bullet-stripped line becomes one entry.

    Args:
        project_lines: Raw lines belonging to the projects section.

    Returns:
        A list of cleaned project description strings.
    """
    return _strip_bullets(project_lines)


def extract_certifications(certification_lines: List[str]) -> List[str]:
    """
    Convert raw lines from the "certifications" section into a list of
    individual certification entries.

    Handles certifications listed comma-separated on a single line, as
    well as one per line.

    Args:
        certification_lines: Raw lines belonging to the certifications
                              section.

    Returns:
        A deduplicated list of cleaned certification strings.
    """
    certifications: List[str] = []

    for line in certification_lines:
        # Certifications are sometimes comma-separated on one line,
        # but commas can also appear within a single certification
        # name (e.g. "AWS Certified Solutions Architect, Associate").
        # Only split on commas if the line looks like a short list
        # (more than one comma present).
        if line.count(",") > 1:
            parts = re.split(r"[,|•▪·●○]", line)
        else:
            parts = [line]

        for part in parts:
            cleaned = part.strip(BULLET_CHARS)
            if cleaned:
                certifications.append(cleaned)

    seen: set = set()
    unique_certifications: List[str] = []
    for cert in certifications:
        key = cert.lower()
        if key not in seen:
            seen.add(key)
            unique_certifications.append(cert)

    return unique_certifications


# ---------------------------------------------------------------------------
# 5. MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def _empty_structured_resume() -> StructuredResume:
    """
    Build an empty/default structured resume result.

    Returns:
        A StructuredResume dictionary with all expected keys present,
        each mapping to an empty value (empty dict for "contact_info",
        empty lists for all other keys).
    """
    return {
        "contact_info": {"name": "", "email": "", "phone": ""},
        "skills": [],
        "education": [],
        "experience": [],
        "projects": [],
        "certifications": [],
    }


def structure_resume(raw_text: str) -> StructuredResume:
    """
    Convert raw resume text into a structured dictionary.

    This is the main public function of this module. It orchestrates
    text cleaning, contact info extraction, section splitting, and
    field extraction for each section.

    Args:
        raw_text: The raw text of the resume (e.g. as extracted by
                  core/pdf_parser.py from a PDF).

    Returns:
        A dictionary with the following keys, always present:
            - "contact_info": Dict[str, str] with keys "name",
              "email", "phone".
            - "skills": List[str]
            - "education": List[str]
            - "experience": List[str]
            - "projects": List[str]
            - "certifications": List[str]

        If `raw_text` is empty, whitespace-only, or contains no usable
        content after cleaning, all fields are returned in their
        default empty form rather than raising an error.
    """
    if not raw_text or not raw_text.strip():
        logger.warning("structure_resume received empty raw_text; returning empty structure.")
        return _empty_structured_resume()

    text = clean_text(raw_text)
    if not text:
        logger.warning("structure_resume: cleaned text is empty; returning empty structure.")
        return _empty_structured_resume()

    logger.info("Structuring resume text (%d characters).", len(text))

    contact_info = extract_contact_info(text)
    sections = split_into_sections(text)

    structured: StructuredResume = {
        "contact_info": contact_info,
        "skills": extract_skills(sections["skills"]),
        "education": extract_education(sections["education"]),
        "experience": extract_experience(sections["experience"]),
        "projects": extract_projects(sections["projects"]),
        "certifications": extract_certifications(sections["certifications"]),
    }

    logger.info(
        "Resume structuring complete: name=%r, skills=%d, education=%d, "
        "experience=%d, projects=%d, certifications=%d",
        contact_info.get("name", ""),
        len(structured["skills"]),
        len(structured["education"]),
        len(structured["experience"]),
        len(structured["projects"]),
        len(structured["certifications"]),
    )

    return structured


# ---------------------------------------------------------------------------
# Manual test entry point (not used by the Streamlit app directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.DEBUG)

    sample_text = """
    Jane Doe
    jane@example.com | +1 555 123 4567

    Skills
    Python, Machine Learning, FAISS, Streamlit, SQL

    Education
    B.Tech in Computer Science, XYZ University, 2024

    Experience
    Data Science Intern, Acme Corp
    Worked on ML pipelines and NLP models.

    Projects
    Built a resume analyzer using NLP and FAISS.

    Certifications
    AWS Certified Cloud Practitioner
    Coursera: Machine Learning Specialization
    """

    result = structure_resume(sample_text)
    print(json.dumps(result, indent=2))

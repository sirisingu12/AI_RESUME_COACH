"""
pdf_parser.py
--------------
Module responsible for extracting structured information from a resume PDF.

Pipeline:
    1. Read the PDF file and extract raw text using pypdf.
    2. Clean and normalize the extracted text.
    3. Split the text into logical sections based on common resume headers.
    4. Extract structured fields (name, email, phone, skills, education,
       projects, experience) using regex and heuristic rules.

Design notes:
    - All functions are pure and independently testable.
    - Malformed/empty/encrypted PDFs are handled gracefully and never raise
      unhandled exceptions; instead, they return an empty/default structure.
    - No external LLM or network calls are made in this module.
"""

from __future__ import annotations

import re
from typing import Dict, List, Union, Optional, BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


# ---------------------------------------------------------------------------
# Type alias for the final parsed resume structure
# ---------------------------------------------------------------------------
ParsedResume = Dict[str, Union[str, List[str]]]


# ---------------------------------------------------------------------------
# Constants: section header keywords used to split the resume into blocks.
# Keys are the canonical section names; values are alternative header
# phrasings commonly found in resumes (all lowercase for matching).
# ---------------------------------------------------------------------------
SECTION_HEADERS: Dict[str, List[str]] = {
    "skills": ["skills", "technical skills", "core competencies", "key skills"],
    "education": ["education", "academic background", "academic qualifications"],
    "projects": ["projects", "academic projects", "personal projects"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "internships",
    ],
}


# ---------------------------------------------------------------------------
# 1. PDF TEXT EXTRACTION
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file: Union[str, BinaryIO]) -> str:
    """
    Extract raw text from a PDF file using pypdf.

    Args:
        file: A file path (str) or a file-like object (e.g. from
              Streamlit's file_uploader) pointing to the PDF.

    Returns:
        The concatenated text of all pages as a single string.
        Returns an empty string if the PDF cannot be read, is encrypted
        without a usable password, or contains no extractable text.
    """
    try:
        reader = PdfReader(file)
    except (PdfReadError, OSError, ValueError):
        # Malformed file, unsupported format, or unreadable stream.
        return ""

    # Handle encrypted PDFs: attempt to decrypt with an empty password,
    # which works for many "owner password only" protected files.
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            # Cannot decrypt -> treat as unreadable.
            return ""

    pages_text: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            # Some malformed pages may raise during extraction;
            # skip that page rather than failing the whole document.
            text = ""
        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


# ---------------------------------------------------------------------------
# 2. TEXT CLEANING
# ---------------------------------------------------------------------------
def clean_text(raw_text: str) -> str:
    """
    Normalize raw extracted text for easier downstream parsing.

    - Collapses multiple spaces/tabs into a single space.
    - Removes trailing whitespace from each line.
    - Removes blank lines (more than one consecutive newline collapsed to one).
    - Strips leading/trailing whitespace from the whole text.

    Args:
        raw_text: Text as extracted directly from the PDF.

    Returns:
        Cleaned text, ready for line-based and regex-based processing.
    """
    if not raw_text:
        return ""

    # Replace tabs and multiple spaces with a single space.
    text = re.sub(r"[ \t]+", " ", raw_text)

    # Strip trailing whitespace on each line.
    lines = [line.strip() for line in text.splitlines()]

    # Remove empty lines.
    lines = [line for line in lines if line]

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 3. CONTACT INFO EXTRACTION
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
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    """
    Extract the first phone-number-like sequence from the text.

    Supports common formats: optional country code, separators of
    space/dot/dash, and 10-digit numbers.

    Args:
        text: Cleaned resume text.

    Returns:
        The phone number as a string, or "" if none is found.
    """
    # Matches patterns like: +91 9876543210, (123) 456-7890, 123-456-7890, etc.
    pattern = r"(\+?\d{1,3}[\s-]?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}"
    for line in text.splitlines():
        match = re.search(pattern, line)
        if match:
            candidate = match.group(0).strip()
            # Require at least 10 digits to avoid matching dates, years, etc.
            digit_count = sum(ch.isdigit() for ch in candidate)
            if digit_count >= 10:
                return candidate
    return ""


def extract_name(text: str) -> str:
    """
    Heuristically extract the candidate's name.

    Assumption: the name is typically the first non-empty line of the
    resume that does not look like an email, phone number, or a
    section header.

    Args:
        text: Cleaned resume text.

    Returns:
        The best-guess candidate name, or "" if none can be determined.
    """
    lines = text.splitlines()
    header_keywords = {kw for kws in SECTION_HEADERS.values() for kw in kws}

    for line in lines[:5]:  # Only check the first few lines.
        candidate = line.strip()
        lower_candidate = candidate.lower()

        if not candidate:
            continue
        if "@" in candidate:  # Likely an email line.
            continue
        if any(ch.isdigit() for ch in candidate):  # Likely a phone/date line.
            continue
        if lower_candidate in header_keywords:  # Likely a section header.
            continue
        if len(candidate.split()) > 6:  # Unlikely to be a name; too long.
            continue

        return candidate

    return ""


# ---------------------------------------------------------------------------
# 4. SECTION SPLITTING
# ---------------------------------------------------------------------------
def split_into_sections(text: str) -> Dict[str, List[str]]:
    """
    Split resume text into named sections based on header keywords.

    The function scans each line; if a line matches one of the known
    section headers (case-insensitive, ignoring punctuation), all
    subsequent lines are assigned to that section until the next
    recognized header is found.

    Args:
        text: Cleaned resume text.

    Returns:
        A dictionary mapping each canonical section name
        ("skills", "education", "projects", "experience") to a list
        of raw lines belonging to that section. Sections not found in
        the resume map to an empty list.
    """
    sections: Dict[str, List[str]] = {key: [] for key in SECTION_HEADERS}
    current_section: Optional[str] = None

    for line in text.splitlines():
        normalized = re.sub(r"[^a-z\s]", "", line.lower()).strip()

        matched_section = _match_section_header(normalized)
        if matched_section:
            current_section = matched_section
            continue  # Don't include the header line itself.

        if current_section:
            sections[current_section].append(line.strip())

    return sections


def _match_section_header(normalized_line: str) -> Optional[str]:
    """
    Check whether a normalized line corresponds to a known section header.

    Args:
        normalized_line: A line of text lowercased and stripped of
                          punctuation/digits.

    Returns:
        The canonical section name if the line matches a known header
        (and is short enough to plausibly BE a header, not body text),
        otherwise None.
    """
    # Headers are typically short (a handful of words).
    if not normalized_line or len(normalized_line.split()) > 4:
        return None

    for section_name, keywords in SECTION_HEADERS.items():
        if normalized_line in keywords:
            return section_name

    return None


# ---------------------------------------------------------------------------
# 5. FIELD EXTRACTION FROM SECTIONS
# ---------------------------------------------------------------------------
def extract_skills(skill_lines: List[str]) -> List[str]:
    """
    Convert raw lines from the "skills" section into a flat list of
    individual skills.

    Handles skills listed comma-separated, separated by pipes/slashes,
    or one per line.

    Args:
        skill_lines: Raw lines belonging to the skills section.

    Returns:
        A deduplicated list of skill strings, trimmed of whitespace.
    """
    skills: List[str] = []

    for line in skill_lines:
        # Split on common delimiters used to list skills inline.
        parts = re.split(r"[,|/•▪·]", line)
        for part in parts:
            cleaned = part.strip(" -:\t")
            if cleaned:
                skills.append(cleaned)

    # Deduplicate while preserving order.
    seen = set()
    unique_skills = []
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

    Each non-empty line is treated as a separate entry (degree,
    institution, year, etc. are kept together as written).

    Args:
        education_lines: Raw lines belonging to the education section.

    Returns:
        A list of cleaned education entry strings.
    """
    return [line.strip(" -:\t•▪") for line in education_lines if line.strip()]


def extract_projects(project_lines: List[str]) -> List[str]:
    """
    Convert raw lines from the "projects" section into a list of
    individual project entries.

    Bullet markers are stripped; each remaining non-empty line becomes
    one entry.

    Args:
        project_lines: Raw lines belonging to the projects section.

    Returns:
        A list of cleaned project description strings.
    """
    return [line.strip(" -:\t•▪") for line in project_lines if line.strip()]


def extract_experience(experience_lines: List[str]) -> List[str]:
    """
    Convert raw lines from the "experience" section into a list of
    individual experience entries.

    Bullet markers are stripped; each remaining non-empty line becomes
    one entry (e.g. job title, company, dates, or a responsibility
    bullet point).

    Args:
        experience_lines: Raw lines belonging to the experience section.

    Returns:
        A list of cleaned experience entry strings.
    """
    return [line.strip(" -:\t•▪") for line in experience_lines if line.strip()]


# ---------------------------------------------------------------------------
# 6. MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def parse_resume(file: Union[str, BinaryIO]) -> ParsedResume:
    """
    Parse a resume PDF into a structured dictionary.

    This is the main public function of this module. It orchestrates
    text extraction, cleaning, section splitting, and field extraction.

    Args:
        file: A file path (str) or file-like object pointing to the
              resume PDF (e.g. from Streamlit's file_uploader).

    Returns:
        A dictionary with the following keys, always present:
            - "name": str
            - "email": str
            - "phone": str
            - "skills": List[str]
            - "education": List[str]
            - "projects": List[str]
            - "experience": List[str]

        If the PDF is empty, malformed, or contains no extractable
        text, all fields are returned in their default empty form
        (empty strings / empty lists) rather than raising an error.
    """
    # Default/empty result structure, returned in all failure cases.
    result: ParsedResume = {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "education": [],
        "projects": [],
        "experience": [],
    }

    raw_text = extract_text_from_pdf(file)
    if not raw_text:
        # Empty or unreadable PDF -> return default structure as-is.
        return result

    text = clean_text(raw_text)
    if not text:
        return result

    # Contact info extraction works on the full cleaned text.
    result["name"] = extract_name(text)
    result["email"] = extract_email(text)
    result["phone"] = extract_phone(text)

    # Split remaining content into sections and extract structured fields.
    sections = split_into_sections(text)
    result["skills"] = extract_skills(sections["skills"])
    result["education"] = extract_education(sections["education"])
    result["projects"] = extract_projects(sections["projects"])
    result["experience"] = extract_experience(sections["experience"])

    return result


# ---------------------------------------------------------------------------
# Manual test entry point (not used by the Streamlit app directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser.py <path_to_resume.pdf>")
        sys.exit(1)

    parsed = parse_resume(sys.argv[1])
    import json

    print(json.dumps(parsed, indent=2))

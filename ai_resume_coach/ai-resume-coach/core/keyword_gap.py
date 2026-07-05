"""
core/keyword_gap.py
--------------------
Module responsible for comparing resume keywords against job
description (JD) keywords and identifying which JD keywords are
present in, or missing from, the resume.

This module is purely text/heuristic based — no LLM, embeddings, or
network calls are involved, making it fast, deterministic, and cheap
to run on every page load.

Future compatibility:
    - jd_matcher.py: can reuse extract_keywords() for consistent
      keyword extraction alongside semantic matching.
    - ats_scorer.py: can reuse analyze_keyword_gap() output (or
      extract_keywords()) when computing the skills/keyword portion of
      the ATS score, ensuring both modules agree on what counts as a
      "keyword".
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set, Union


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
KeywordGapResult = Dict[str, Union[List[str], int, float]]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A small, general-purpose stopword list. Removing these prevents
# common words (e.g. "the", "and", "with") and generic resume/JD
# filler words from being treated as meaningful keywords.
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
    "team", "across", "within", "ensure", "ensuring", "candidate",
    "candidates", "looking", "required", "requirements", "preferred",
    "plus", "good", "excellent", "communication",
}

# Minimum length (in characters) for a single-word token to be
# considered a meaningful keyword.
MIN_KEYWORD_LENGTH: int = 2

# Maximum number of words a chunk can have to be treated as a single
# candidate phrase (e.g. "machine learning"). Longer chunks are
# treated as sentences and broken into individual word tokens.
MAX_PHRASE_WORDS: int = 4

# Delimiters used to split text into candidate keyword/phrase chunks
# (commas, semicolons, bullets, pipes, slashes).
CHUNK_DELIMITERS: str = r"[,;|•▪·●○/]"


# ---------------------------------------------------------------------------
# 1. KEYWORD EXTRACTION
# ---------------------------------------------------------------------------
def extract_keywords(text: str) -> List[str]:
    """
    Extract candidate keyword/phrase tokens from free text.

    Approach (heuristic, no external NLP library required):
        - Lowercase the text.
        - Split into lines, then into delimiter-separated chunks
          (resumes and JDs often list skills/requirements this way).
        - Chunks of up to MAX_PHRASE_WORDS words are kept as single
          candidate phrases (e.g. "machine learning", "rest apis").
        - Longer chunks (full sentences) are broken into individual
          word tokens instead.
        - Stopwords, very short tokens, and duplicates are removed.

    Args:
        text: Raw text (resume or job description).

    Returns:
        A deduplicated list of candidate keyword/phrase strings,
        preserving first-seen order. Returns an empty list if `text`
        is empty or whitespace-only.
    """
    if not text or not text.strip():
        return []

    lowered = text.lower()
    keywords: List[str] = []
    seen: Set[str] = set()

    for line in lowered.splitlines():
        for chunk in re.split(CHUNK_DELIMITERS, line):
            chunk = chunk.strip(" -:\t()[]")
            if not chunk:
                continue

            words = chunk.split()

            if 1 <= len(words) <= MAX_PHRASE_WORDS:
                candidate = _clean_token(chunk)
                _add_keyword(candidate, keywords, seen)
            else:
                for word in words:
                    candidate = _clean_token(word)
                    _add_keyword(candidate, keywords, seen)

    logger.debug("Extracted %d keywords from text of length %d.", len(keywords), len(text))
    return keywords


def _clean_token(token: str) -> str:
    """
    Normalize a candidate keyword/phrase: strip characters that aren't
    letters, digits, spaces, or the symbols '+', '#', '.' (which can be
    meaningful in tech terms like "c++", "c#", "node.js"), and trim
    surrounding whitespace/dots.

    Args:
        token: A raw candidate phrase or word.

    Returns:
        The cleaned token, or "" if nothing meaningful remains.
    """
    cleaned = re.sub(r"[^a-z0-9+#. ]", "", token).strip(" .")
    return cleaned


def _add_keyword(candidate: str, keywords: List[str], seen: Set[str]) -> None:
    """
    Add a candidate keyword to the result list if it passes filters
    (non-empty, long enough, not a stopword) and hasn't been seen yet.

    Args:
        candidate: The cleaned candidate keyword/phrase.
        keywords: The list to append accepted keywords to (mutated in
                  place).
        seen: A set tracking already-added keywords (mutated in
              place).
    """
    if not candidate or len(candidate) < MIN_KEYWORD_LENGTH:
        return

    words = candidate.split()

    # Single-word candidates: skip if it's a stopword.
    if len(words) == 1 and candidate in STOPWORDS:
        return

    # Multi-word candidates: skip only if EVERY word is a stopword
    # (e.g. "and the" would otherwise slip through).
    if len(words) > 1 and all(word in STOPWORDS for word in words):
        return

    if candidate not in seen:
        seen.add(candidate)
        keywords.append(candidate)


# ---------------------------------------------------------------------------
# 2. PRESENCE MATCHING
# ---------------------------------------------------------------------------
def _is_present(keyword: str, resume_keyword_set: Set[str]) -> bool:
    """
    Check whether a JD keyword/phrase is "present" in the resume's
    keyword set.

    A keyword is considered present if:
        - It appears verbatim in `resume_keyword_set`, or
        - It is a substring of any resume keyword, or any resume
          keyword is a substring of it (handles cases like
          JD="python" vs resume="python3", or JD="rest api" vs
          resume="rest apis").

    Args:
        keyword: A single keyword/phrase from the JD.
        resume_keyword_set: The set of keywords/phrases extracted from
                             the resume.

    Returns:
        True if a match is found, False otherwise.
    """
    if keyword in resume_keyword_set:
        return True

    return any(
        keyword in resume_kw or resume_kw in keyword
        for resume_kw in resume_keyword_set
    )


# ---------------------------------------------------------------------------
# 3. MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def analyze_keyword_gap(resume_text: str, jd_text: str) -> KeywordGapResult:
    """
    Compare resume text against job description text and identify
    which JD keywords are present in, or missing from, the resume.

    Args:
        resume_text: The full text of the resume (e.g. all parsed
                      sections joined together).
        jd_text: The full text of the job description.

    Returns:
        A dictionary with the following keys:
            - "present_keywords": List[str] — JD keywords/phrases
              found in the resume (exact or substring match),
              preserving the order they appear in the JD.
            - "missing_keywords": List[str] — JD keywords/phrases not
              found in the resume, preserving the order they appear
              in the JD.
            - "total_keywords": int — total number of distinct JD
              keywords considered (len(present) + len(missing)).
            - "match_percentage": float — percentage (0-100, rounded
              to 2 decimals) of JD keywords found in the resume.

        If `jd_text` is empty/whitespace, all lists are empty,
        "total_keywords" is 0, and "match_percentage" is 0.0.
    """
    jd_keywords = extract_keywords(jd_text)

    if not jd_keywords:
        logger.warning("analyze_keyword_gap: job description produced no keywords.")
        return {
            "present_keywords": [],
            "missing_keywords": [],
            "total_keywords": 0,
            "match_percentage": 0.0,
        }

    resume_keywords = extract_keywords(resume_text)
    resume_keyword_set = set(resume_keywords)

    if not resume_keyword_set:
        logger.info("analyze_keyword_gap: resume produced no keywords; all JD keywords are missing.")

    present_keywords: List[str] = []
    missing_keywords: List[str] = []

    for jd_keyword in jd_keywords:
        if resume_keyword_set and _is_present(jd_keyword, resume_keyword_set):
            present_keywords.append(jd_keyword)
        else:
            missing_keywords.append(jd_keyword)

    total_keywords = len(jd_keywords)
    match_percentage = round((len(present_keywords) / total_keywords) * 100, 2)

    logger.info(
        "Keyword gap analysis: %d/%d JD keywords present (%.2f%%).",
        len(present_keywords),
        total_keywords,
        match_percentage,
    )

    return {
        "present_keywords": present_keywords,
        "missing_keywords": missing_keywords,
        "total_keywords": total_keywords,
        "match_percentage": match_percentage,
    }


# ---------------------------------------------------------------------------
# Manual test entry point (not used by the Streamlit app directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.DEBUG)

    sample_resume_text = (
        "Skills: Python, Machine Learning, FAISS, Streamlit, SQL\n"
        "Experience: Data Science Intern at Acme Corp, worked on ML "
        "pipelines and NLP models.\n"
        "Projects: Built a resume analyzer using NLP and vector search."
    )

    sample_jd_text = (
        "We are looking for a Python developer with experience in "
        "machine learning, natural language processing, FAISS, and "
        "Streamlit. Familiarity with Docker and Kubernetes is a plus."
    )

    result = analyze_keyword_gap(sample_resume_text, sample_jd_text)
    print(json.dumps(result, indent=2))

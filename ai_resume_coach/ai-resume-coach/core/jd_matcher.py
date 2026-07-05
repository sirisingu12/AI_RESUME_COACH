"""
jd_matcher.py
--------------
Module responsible for semantically comparing a resume against a job
description (JD) using sentence-transformers embeddings.

Tasks performed:
    1. Semantic similarity score — overall cosine similarity between
       the resume text and JD text embeddings, scaled to 0-100.
    2. Keyword extraction — pull out candidate "skill-like" keywords
       from the JD (and resume) using simple NLP heuristics.
    3. Matching skill detection — JD keywords that are present in the
       resume, either as exact text matches or as semantically similar
       phrases (via embedding similarity).
    4. Missing skill detection — JD keywords with no exact or semantic
       match anywhere in the resume.

Model:
    - sentence-transformers/all-MiniLM-L6-v2 (small, fast, runs fully
      locally — no network calls after the model is cached).

This module performs no LLM calls and no network access at inference
time (aside from the one-time model download/cache by
sentence-transformers on first use).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from config import EMBEDDING_MODEL, SEMANTIC_MATCH_THRESHOLD as CONFIG_SEMANTIC_MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
MatchResult = Dict[str, object]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Name of the local sentence-transformers model used for all embeddings.
# Sourced from config.py so it stays in sync with embeddings.py.
MODEL_NAME: str = EMBEDDING_MODEL

# Stopwords removed during keyword extraction. Kept small and general
# so it works across resumes/JDs from different domains.
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

# Minimum token length to be considered a meaningful keyword.
MIN_KEYWORD_LENGTH: int = 2

# Semantic similarity threshold above which two phrases are considered
# a "match" even if their exact text differs (e.g. "ML" vs "machine
# learning"). Sourced from config.py so the threshold can be tuned in
# one place (also used by suggestion/keyword-gap features if needed).
SEMANTIC_MATCH_THRESHOLD: float = CONFIG_SEMANTIC_MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# 1. MODEL LOADING (cached, loaded once per process)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Load (and cache) the sentence-transformers model.

    Using lru_cache ensures the model is loaded from disk only once per
    process, regardless of how many times this function is called —
    important for Streamlit, which may re-run scripts frequently.

    Returns:
        A loaded SentenceTransformer instance for MODEL_NAME.
    """
    return SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------------
# 2. SEMANTIC SIMILARITY SCORE
# ---------------------------------------------------------------------------
def calculate_semantic_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute the overall semantic similarity between the resume and the
    job description.

    Logic:
        - Embed both texts using the sentence-transformers model.
        - Compute cosine similarity between the two embeddings
          (range: -1 to 1, but for normal text typically 0 to 1).
        - Scale the result to a 0-100 "match score" for readability.

    Args:
        resume_text: The full text of the resume (e.g. all sections
                      joined together).
        jd_text: The full text of the job description.

    Returns:
        A match score between 0 and 100, rounded to 2 decimal places.
        Returns 0.0 if either input is empty.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    model = get_model()
    embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)

    similarity = cos_sim(embeddings[0], embeddings[1]).item()

    # Clamp to [0, 1] in case of minor floating point negatives, then
    # scale to a 0-100 score.
    similarity = max(0.0, min(1.0, similarity))

    return round(similarity * 100, 2)


# ---------------------------------------------------------------------------
# 3. KEYWORD EXTRACTION
# ---------------------------------------------------------------------------
def extract_keywords(text: str) -> List[str]:
    """
    Extract candidate "skill-like" keywords/phrases from free text.

    Approach (heuristic, no external NLP library required):
        - Lowercase the text.
        - Split into lines, then into comma/bullet/pipe-separated chunks
          (resumes and JDs often list skills this way).
        - For chunks that look like a single phrase (<= 4 words), keep
          them as candidate keyword phrases.
        - For longer chunks (full sentences), fall back to extracting
          individual significant word tokens.
        - Remove stopwords, short tokens, and duplicates.

    Args:
        text: Raw text (resume or JD).

    Returns:
        A deduplicated list of candidate keyword/phrase strings,
        preserving first-seen order.
    """
    if not text:
        return []

    lowered = text.lower()
    keywords: List[str] = []
    seen: Set[str] = set()

    for line in lowered.splitlines():
        # Split on common list delimiters used for skills/requirements.
        chunks = re.split(r"[,;|•▪·/]", line)

        for chunk in chunks:
            chunk = chunk.strip(" -:\t()")
            if not chunk:
                continue

            words = chunk.split()

            if 1 <= len(words) <= 4:
                # Treat as a single candidate phrase (e.g. "machine
                # learning", "rest apis", "python").
                candidate = _clean_phrase(chunk)
                _add_keyword(candidate, keywords, seen)
            else:
                # Long sentence: fall back to individual word tokens.
                for word in words:
                    candidate = _clean_phrase(word)
                    _add_keyword(candidate, keywords, seen)

    return keywords


def _clean_phrase(phrase: str) -> str:
    """
    Normalize a candidate keyword phrase: strip punctuation (except
    '+', '#', '.' which can be meaningful in tech terms like "c++",
    "c#", "node.js") and surrounding whitespace.

    Args:
        phrase: A raw candidate phrase or word.

    Returns:
        The cleaned phrase, or "" if nothing meaningful remains.
    """
    cleaned = re.sub(r"[^a-z0-9+#. ]", "", phrase).strip(" .")
    return cleaned


def _add_keyword(candidate: str, keywords: List[str], seen: Set[str]) -> None:
    """
    Add a candidate keyword to the result list if it passes filters
    (non-empty, long enough, not a stopword) and hasn't been seen yet.

    Args:
        candidate: The cleaned candidate keyword/phrase.
        keywords: The list to append accepted keywords to (mutated
                  in place).
        seen: A set tracking already-added keywords (mutated in place).
    """
    if not candidate or len(candidate) < MIN_KEYWORD_LENGTH:
        return

    # For single-word candidates, skip stopwords entirely.
    words = candidate.split()
    if len(words) == 1 and candidate in STOPWORDS:
        return

    # For multi-word candidates, skip if EVERY word is a stopword
    # (e.g. "and the" would otherwise slip through).
    if len(words) > 1 and all(word in STOPWORDS for word in words):
        return

    if candidate not in seen:
        seen.add(candidate)
        keywords.append(candidate)


# ---------------------------------------------------------------------------
# 4. MATCHED / MISSING SKILL DETECTION
# ---------------------------------------------------------------------------
def _is_exact_match(jd_keyword: str, resume_keywords: Set[str]) -> bool:
    """
    Check whether a JD keyword/phrase appears verbatim among the
    resume's extracted keywords, or as a substring of any resume
    keyword (handles cases like JD="python" and resume="python3").

    Args:
        jd_keyword: A single keyword/phrase from the JD.
        resume_keywords: The set of keywords/phrases extracted from
                          the resume.

    Returns:
        True if an exact or substring match is found.
    """
    if jd_keyword in resume_keywords:
        return True

    return any(
        jd_keyword in resume_kw or resume_kw in jd_keyword
        for resume_kw in resume_keywords
    )


def _find_best_semantic_match(
    jd_keyword_embedding: "np.ndarray",
    resume_keyword_embeddings: "np.ndarray",
) -> float:
    """
    Find the highest cosine similarity between a single JD keyword
    embedding and all resume keyword embeddings.

    Args:
        jd_keyword_embedding: Embedding vector for one JD keyword,
                               shape (embedding_dim,).
        resume_keyword_embeddings: Embedding matrix for all resume
                                    keywords, shape
                                    (num_resume_keywords, embedding_dim).

    Returns:
        The maximum cosine similarity score (float), or 0.0 if
        `resume_keyword_embeddings` is empty.
    """
    if resume_keyword_embeddings.shape[0] == 0:
        return 0.0

    similarities = cos_sim(jd_keyword_embedding, resume_keyword_embeddings)
    return float(similarities.max().item())


def detect_matched_and_missing_skills(
    resume_text: str,
    jd_text: str,
) -> Tuple[List[str], List[str]]:
    """
    Detect which JD skills/keywords are present in the resume
    (matched) versus absent (missing), using both exact text matching
    and semantic (embedding-based) similarity.

    Logic:
        1. Extract keyword lists from both the JD and the resume.
        2. For each JD keyword:
            a. If it (or a close substring) appears in the resume
               keywords -> matched (exact match).
            b. Otherwise, embed the JD keyword and compare it against
               embeddings of all resume keywords. If the best
               similarity exceeds SEMANTIC_MATCH_THRESHOLD -> matched
               (semantic match).
            c. Otherwise -> missing.

    Args:
        resume_text: The full text of the resume.
        jd_text: The full text of the job description.

    Returns:
        A tuple (matched_skills, missing_skills), each a list of
        strings (JD keywords/phrases), preserving the order in which
        they appear in the JD's extracted keyword list.
    """
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)

    if not jd_keywords:
        return [], []

    resume_keyword_set = set(resume_keywords)

    matched: List[str] = []
    missing: List[str] = []

    # Pre-embed all resume keywords once for efficient semantic
    # comparisons (avoids re-encoding inside the loop).
    model = get_model()
    resume_embeddings = (
        model.encode(resume_keywords, convert_to_tensor=True)
        if resume_keywords
        else np.empty((0, 0))
    )

    for jd_keyword in jd_keywords:
        if _is_exact_match(jd_keyword, resume_keyword_set):
            matched.append(jd_keyword)
            continue

        if resume_keywords:
            jd_embedding = model.encode(jd_keyword, convert_to_tensor=True)
            best_similarity = _find_best_semantic_match(
                jd_embedding, resume_embeddings
            )
            if best_similarity >= SEMANTIC_MATCH_THRESHOLD:
                matched.append(jd_keyword)
                continue

        missing.append(jd_keyword)

    return matched, missing


# ---------------------------------------------------------------------------
# 5. MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def match_resume_to_job(resume_text: str, jd_text: str) -> MatchResult:
    """
    Compare a resume against a job description and produce a full
    match report.

    Args:
        resume_text: The full text of the resume (e.g. all parsed
                      sections joined together).
        jd_text: The full text of the job description.

    Returns:
        A dictionary with the following keys:
            - "match_score": float, 0-100. Overall semantic similarity
              between the resume and JD.
            - "matched_skills": List[str]. JD keywords/phrases found
              in the resume (exact or semantic match).
            - "missing_skills": List[str]. JD keywords/phrases not
              found in the resume.

        If either input is empty, returns a zeroed-out result with
        empty skill lists.
    """
    match_score = calculate_semantic_similarity(resume_text, jd_text)
    matched_skills, missing_skills = detect_matched_and_missing_skills(
        resume_text, jd_text
    )

    return {
        "match_score": match_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }


# ---------------------------------------------------------------------------
# Manual test entry point (not used by the Streamlit app directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
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

    import json

    print(json.dumps(match_resume_to_job(sample_resume_text, sample_jd_text), indent=2))

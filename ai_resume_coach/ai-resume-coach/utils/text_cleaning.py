"""
utils/text_cleaning.py
-----------------------
Text normalization, tokenization, and formatting helpers shared
across the entire AI Resume Analyzer & Career Coach project.

These are stateless, pure utility functions — no LLM calls, no I/O,
no side effects. Every function is independently testable and safe to
import from any module (core/, ui/, or other utils/).

Why this exists alongside core/resume_structurer.py:
    - resume_structurer.py owns domain-specific parsing logic
      (section splitting, field extraction, contact info detection).
    - This module owns general-purpose text utilities that are reused
      in multiple places: the UI display layer, prompt construction
      in llm_client.py wrappers, keyword post-processing, etc.
    - Keeping them separate prevents circular imports and keeps each
      module focused.

Public API:
    - clean_whitespace()        : collapse whitespace, strip blank lines
    - normalize_text()          : lowercase, strip punctuation
    - truncate_text()           : hard-cap text for LLM prompt safety
    - split_into_sentences()    : naive sentence tokenizer
    - remove_stopwords()        : filter a token list
    - extract_alpha_tokens()    : tokenize to letter-only words
    - format_list_as_bullets()  : List[str] -> markdown bullet string
    - format_resume_for_llm()   : StructuredResume dict -> flat prompt text
    - highlight_keywords()      : wrap matched keywords in markdown bold
    - truncate_for_display()    : shorten text for UI cards/previews
    - count_words()             : word count for progress/info display
    - is_meaningful_text()      : guard against empty/junk input
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# General-purpose stopwords used by remove_stopwords() and helpers.
# Intentionally kept small and domain-agnostic; domain-specific lists
# live in ats_scorer.py and keyword_gap.py.
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on",
    "for", "with", "at", "by", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "i", "me", "my", "we", "you", "your", "our", "their", "they", "he",
    "she", "him", "her", "will", "shall", "can", "may", "should", "would",
    "must", "have", "has", "had", "do", "does", "did", "not", "no", "so",
    "such", "than", "into", "about", "etc",
}

# Characters stripped from token edges when cleaning individual words.
_PUNCT_STRIP_CHARS: str = " .,;:!?\"'()-–—[]{}/*\\"

# Maximum characters shown in a UI preview card before truncation.
_DEFAULT_PREVIEW_LENGTH: int = 250

# Hard limit for text sent in LLM prompts, in characters.
# ~8 000 chars ≈ 2 000 tokens, which fits most local models comfortably.
_DEFAULT_LLM_CHAR_LIMIT: int = 8_000


# ---------------------------------------------------------------------------
# 1. WHITESPACE & BASIC NORMALIZATION
# ---------------------------------------------------------------------------

def clean_whitespace(text: str) -> str:
    """
    Collapse redundant whitespace and remove blank lines.

    Steps:
        - Replace tabs and multiple spaces with a single space.
        - Strip trailing/leading whitespace from each line.
        - Drop blank lines (including lines that are only whitespace).
        - Strip leading/trailing whitespace from the result.

    This is equivalent to core/resume_structurer.clean_text() but
    lives here so that UI and utils code can call it without importing
    from the core package.

    Args:
        text: Any raw string (PDF-extracted text, user input, etc.)

    Returns:
        Cleaned string, or "" if input is empty/whitespace-only.

    Examples:
        >>> clean_whitespace("  hello   world  \n\n  foo  ")
        'hello world\nfoo'
    """
    if not text or not text.strip():
        return ""

    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def normalize_text(text: str, *, lowercase: bool = True, remove_punct: bool = True) -> str:
    """
    Normalize text for keyword comparison or NLP preprocessing.

    Steps (all optional via kwargs):
        - Lowercase.
        - Remove punctuation (keeps letters, digits, spaces, and
          tech-friendly symbols: '+', '#', '.').
        - Collapse resulting multiple spaces.
        - Strip leading/trailing whitespace.

    Args:
        text: The input string to normalize.
        lowercase: Whether to convert to lowercase (default True).
        remove_punct: Whether to strip punctuation (default True).

    Returns:
        Normalized string. Returns "" for empty/whitespace input.

    Examples:
        >>> normalize_text("Python, Machine Learning; REST APIs!")
        'python machine learning rest apis'
    """
    if not text or not text.strip():
        return ""

    result = text

    if lowercase:
        result = result.lower()

    if remove_punct:
        # Keep letters, digits, spaces, and +/#/. (tech symbols).
        result = re.sub(r"[^a-z0-9+#.\s]", " ", result)

    result = re.sub(r"\s+", " ", result).strip()
    return result


def normalize_unicode(text: str) -> str:
    """
    Convert non-ASCII Unicode characters to their closest ASCII
    equivalents where possible (e.g. 'é' → 'e', '–' → '-').

    Useful as a preprocessing step before regex-based parsing on
    text extracted from PDFs that may use Unicode bullets, dashes,
    and accented characters.

    Args:
        text: Input string, possibly containing non-ASCII characters.

    Returns:
        ASCII-safe string. Characters with no ASCII equivalent are
        dropped.

    Examples:
        >>> normalize_unicode("Résumé — Software Éngineer")
        'Resume - Software Engineer'
    """
    if not text:
        return ""

    # Decompose Unicode characters (NFD) so that accents become
    # separate combining characters, then encode to ASCII dropping
    # anything that cannot be represented.
    normalized = unicodedata.normalize("NFD", text)
    ascii_bytes = normalized.encode("ascii", errors="ignore")
    result = ascii_bytes.decode("ascii")

    # Normalize dashes: em dash, en dash → hyphen.
    result = result.replace("\u2014", "-").replace("\u2013", "-")

    return result


# ---------------------------------------------------------------------------
# 2. TRUNCATION
# ---------------------------------------------------------------------------

def truncate_text(
    text: str,
    max_chars: int = _DEFAULT_LLM_CHAR_LIMIT,
    suffix: str = "... [truncated]",
) -> str:
    """
    Hard-cap text at `max_chars` characters.

    Used before inserting resume text or job descriptions into LLM
    prompts to avoid exceeding context window limits.

    The cut is made at a word boundary where possible (searching
    backwards from max_chars for the last space), so the truncation
    point never splits a word in the middle.

    Args:
        text: The text to truncate.
        max_chars: Maximum number of characters to allow.
        suffix: String appended when truncation occurs.

    Returns:
        The original text if it fits within `max_chars`, otherwise the
        truncated text with `suffix` appended. Returns "" for empty
        input.

    Examples:
        >>> truncate_text("hello world", max_chars=7, suffix="...")
        'hello...'
    """
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    # Trim to max_chars minus room for the suffix.
    cut_at = max_chars - len(suffix)
    if cut_at <= 0:
        return suffix[:max_chars]

    # Walk back to the nearest word boundary.
    truncated = text[:cut_at]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + suffix


def truncate_for_display(
    text: str,
    max_chars: int = _DEFAULT_PREVIEW_LENGTH,
    suffix: str = "…",
) -> str:
    """
    Shorten text for UI preview cards or one-line summaries.

    Identical logic to truncate_text() but uses a single-char ellipsis
    (…) by default and a shorter default limit, making it suitable for
    Streamlit expanders, card bodies, and tooltips.

    Args:
        text: The text to shorten.
        max_chars: Maximum characters to show (default 250).
        suffix: String appended on truncation (default '…').

    Returns:
        Possibly-truncated string. Returns "" for empty input.
    """
    return truncate_text(text, max_chars=max_chars, suffix=suffix)


# ---------------------------------------------------------------------------
# 3. TOKENIZATION
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into individual sentences using punctuation heuristics.

    Not a full NLP sentence tokenizer — it handles the common cases
    found in resume text and job descriptions (periods, exclamation
    marks, question marks followed by whitespace or end of string).

    Args:
        text: Input text to split.

    Returns:
        List of stripped, non-empty sentence strings. Returns an empty
        list for empty/whitespace-only input.

    Examples:
        >>> split_into_sentences("I love Python. It's great! Really.")
        ['I love Python.', "It's great!", 'Really.']
    """
    if not text or not text.strip():
        return []

    # Split on sentence-ending punctuation followed by whitespace or EOS.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def extract_alpha_tokens(text: str) -> List[str]:
    """
    Tokenize text into purely alphabetic word tokens (no digits,
    punctuation, or symbols), lowercased.

    Useful for stopword filtering, vocabulary analysis, and anywhere
    only plain words matter (as opposed to tech keywords like "c++"
    or "node.js" which require less aggressive cleaning).

    Args:
        text: Any input text.

    Returns:
        List of lowercase alphabetic tokens. Returns [] for empty input.

    Examples:
        >>> extract_alpha_tokens("Led a team of 5 engineers.")
        ['led', 'a', 'team', 'of', 'engineers']
    """
    if not text:
        return []

    return re.findall(r"[a-zA-Z]+", text.lower())


# ---------------------------------------------------------------------------
# 4. STOPWORD FILTERING
# ---------------------------------------------------------------------------

def remove_stopwords(
    tokens: List[str],
    extra_stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Filter a list of tokens, removing common stopwords.

    Args:
        tokens: A list of (ideally lowercased) word strings.
        extra_stopwords: An optional additional set of words to remove
                          on top of the built-in list. Useful for
                          domain-specific filler words.

    Returns:
        A new list with stopwords removed, in the original order.

    Examples:
        >>> remove_stopwords(['i', 'worked', 'on', 'machine', 'learning'])
        ['worked', 'machine', 'learning']
    """
    stop = _STOPWORDS
    if extra_stopwords:
        stop = stop | extra_stopwords

    return [token for token in tokens if token.lower() not in stop]


# ---------------------------------------------------------------------------
# 5. FORMATTING FOR LLM PROMPTS
# ---------------------------------------------------------------------------

def format_resume_for_llm(
    structured_resume: Dict[str, Union[Dict, List[str]]],
    max_chars: int = _DEFAULT_LLM_CHAR_LIMIT,
) -> str:
    """
    Convert a structured resume dictionary into a clean, human-readable
    text block suitable for inclusion in LLM prompts.

    Handles both schemas produced by the core modules:
        - resume_structurer.structure_resume() → nested contact_info
        - pdf_parser.parse_resume()            → flat name/email/phone

    The output is a plain-text block with labelled sections, formatted
    so the LLM can parse it naturally without needing JSON context.

    Args:
        structured_resume: The structured resume dict from either
            core/resume_structurer.py or core/pdf_parser.py.
        max_chars: Hard character limit applied after formatting.
            Defaults to _DEFAULT_LLM_CHAR_LIMIT (8 000 chars).

    Returns:
        A formatted string with section labels and bullet-style entries,
        truncated to `max_chars` if necessary.

    Examples:
        >>> r = {"name": "Jane", "email": "j@x.com", "phone": "123",
        ...      "skills": ["Python", "SQL"], "education": ["B.Tech 2024"],
        ...      "experience": ["DS Intern @ Acme"], "projects": []}
        >>> print(format_resume_for_llm(r))
        Name: Jane
        Email: j@x.com
        Phone: 123
        ...
    """
    if not structured_resume:
        return ""

    lines: List[str] = []

    # -- Contact info -------------------------------------------------------
    # Handle both the flat schema (pdf_parser) and nested schema
    # (resume_structurer) transparently.
    contact = structured_resume.get("contact_info")

    if isinstance(contact, dict):
        name  = contact.get("name", "")
        email = contact.get("email", "")
        phone = contact.get("phone", "")
    else:
        # Flat schema from pdf_parser.
        name  = str(structured_resume.get("name", ""))
        email = str(structured_resume.get("email", ""))
        phone = str(structured_resume.get("phone", ""))

    if name:
        lines.append(f"Name: {name}")
    if email:
        lines.append(f"Email: {email}")
    if phone:
        lines.append(f"Phone: {phone}")

    # -- Sections -----------------------------------------------------------
    section_labels: List[tuple] = [
        ("skills",           "Skills"),
        ("experience",       "Experience"),
        ("projects",         "Projects"),
        ("education",        "Education"),
        ("certifications",   "Certifications"),
    ]

    for key, label in section_labels:
        items = structured_resume.get(key)
        if not items:
            continue

        if isinstance(items, list):
            # Filter out any empty strings that may have slipped through.
            items = [str(i).strip() for i in items if str(i).strip()]

        if not items:
            continue

        lines.append(f"\n{label}:")
        for item in items:
            lines.append(f"  • {item}")

    full_text = "\n".join(lines).strip()
    return truncate_text(full_text, max_chars=max_chars)


def format_list_as_bullets(
    items: List[str],
    bullet: str = "•",
    indent: str = "",
) -> str:
    """
    Convert a list of strings into a markdown/plain-text bullet list.

    Used by ui/ pages to render LLM-generated lists (strengths,
    suggestions, roadmap stages, etc.) consistently.

    Args:
        items: The list of strings to format.
        bullet: The bullet character to use (default "•").
        indent: Optional leading whitespace before each line (default "").

    Returns:
        A multi-line string with one bullet per line. Returns ""
        if `items` is empty.

    Examples:
        >>> format_list_as_bullets(["Write tests", "Add type hints"])
        '• Write tests\n• Add type hints'
    """
    if not items:
        return ""

    lines = [f"{indent}{bullet} {item.strip()}" for item in items if item.strip()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. KEYWORD HIGHLIGHTING
# ---------------------------------------------------------------------------

def highlight_keywords(
    text: str,
    keywords: List[str],
    template: str = "**{word}**",
) -> str:
    """
    Wrap occurrences of `keywords` in `text` with a formatting template.

    Matching is case-insensitive and whole-word only, so "Python" will
    match "python" but not "pythonista".

    The default template wraps matches in markdown bold (**word**),
    which renders correctly in Streamlit's st.markdown().

    Args:
        text: The source text in which to highlight keywords.
        keywords: A list of keyword strings to look for.
        template: A format string with a single {word} placeholder
                   (default "**{word}**" for markdown bold).

    Returns:
        The text with matched keywords replaced by the formatted
        version. Returns `text` unchanged if `keywords` is empty.

    Examples:
        >>> highlight_keywords("Know Python and SQL", ["Python", "SQL"])
        'Know **Python** and **SQL**'
    """
    if not text or not keywords:
        return text

    result = text

    # Sort longest keywords first so that multi-word phrases like
    # "machine learning" are matched before "machine" alone.
    sorted_kws = sorted(keywords, key=len, reverse=True)

    for kw in sorted_kws:
        if not kw.strip():
            continue

        # Escape any regex special characters in the keyword itself.
        escaped = re.escape(kw.strip())

        # Whole-word match, case-insensitive.
        pattern = rf"\b({escaped})\b"

        replacement = template.format(word=r"\1")

        try:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        except re.error:
            # If the keyword produces an invalid pattern for any reason,
            # skip it silently rather than crashing the UI.
            continue

    return result


# ---------------------------------------------------------------------------
# 7. MISCELLANEOUS HELPERS
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """
    Count the number of whitespace-separated word tokens in `text`.

    Args:
        text: Any string.

    Returns:
        Integer word count. Returns 0 for empty/whitespace-only input.

    Examples:
        >>> count_words("I have 5 years of Python experience.")
        7
    """
    if not text or not text.strip():
        return 0

    return len(text.split())


def is_meaningful_text(text: str, min_words: int = 10) -> bool:
    """
    Guard function: check whether `text` contains enough content to
    be worth sending to the LLM or scoring engine.

    Used at the top of UI page render functions to show a friendly
    "please upload a resume first" message instead of sending an
    empty or near-empty string to the backend.

    Args:
        text: The text to evaluate (resume text, job description, etc.)
        min_words: Minimum word count for the text to be considered
                    meaningful (default 10).

    Returns:
        True if `text` is non-empty and contains at least `min_words`
        whitespace-separated words, False otherwise.

    Examples:
        >>> is_meaningful_text("Hello world", min_words=3)
        False
        >>> is_meaningful_text("I have experience in Python and data science tools", min_words=5)
        True
    """
    return count_words(text) >= min_words


def slugify(text: str) -> str:
    """
    Convert a string to a lowercase, hyphen-separated slug.

    Useful for generating session_state keys, file names, or HTML ids
    from human-readable labels like page names.

    Args:
        text: Input string (e.g. "ATS Score", "📊 ATS Score").

    Returns:
        A lowercase slug with non-alphanumeric characters replaced by
        hyphens, with no leading/trailing hyphens.

    Examples:
        >>> slugify("📊 ATS Score")
        'ats-score'
        >>> slugify("Job Description Match")
        'job-description-match'
    """
    if not text:
        return ""

    # Strip emoji and non-ASCII.
    ascii_text = text.encode("ascii", errors="ignore").decode("ascii")

    # Lowercase and replace non-alphanumeric runs with a hyphen.
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower())
    return slug.strip("-")


# ---------------------------------------------------------------------------
# Manual test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Jane Doe\n"
        "jane@example.com  |  +1 555 123 4567\n\n"
        "Skills:  Python,  Machine Learning,  FAISS,  Streamlit\n\n"
        "Experience:\n"
        "• Data Science Intern, Acme Corp — worked on ML pipelines.\n\n"
        "Education:\n"
        "B.Tech in Computer Science, 2024\n"
    )

    print("=== clean_whitespace ===")
    print(clean_whitespace(sample))

    print("\n=== normalize_text ===")
    print(normalize_text("Python, Machine Learning; REST APIs!"))

    print("\n=== truncate_text ===")
    print(truncate_text(sample, max_chars=80))

    print("\n=== split_into_sentences ===")
    print(split_into_sentences("I built a chatbot. It used FAISS. Really cool!"))

    print("\n=== remove_stopwords ===")
    tokens = extract_alpha_tokens("I worked on machine learning projects")
    print(remove_stopwords(tokens))

    print("\n=== format_resume_for_llm ===")
    structured = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "+1 555 123 4567",
        "skills": ["Python", "FAISS", "Streamlit"],
        "education": ["B.Tech CS 2024"],
        "experience": ["DS Intern, Acme Corp"],
        "projects": ["Resume Analyzer"],
    }
    print(format_resume_for_llm(structured))

    print("\n=== highlight_keywords ===")
    print(highlight_keywords("Know Python and SQL well", ["Python", "SQL"]))

    print("\n=== is_meaningful_text ===")
    print(is_meaningful_text("hi", min_words=5))           # False
    print(is_meaningful_text(sample, min_words=5))          # True

    print("\n=== slugify ===")
    print(slugify("📊 ATS Score"))                          # ats-score
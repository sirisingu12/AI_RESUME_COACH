"""
utils/session_state.py
-----------------------
Centralized helpers for managing Streamlit's st.session_state across
all pages of the AI Resume Analyzer & Career Coach.

Why this exists:
    - Streamlit re-runs the entire script on every user interaction.
      session_state is the only safe place to persist data between
      re-runs (uploaded resume, parsed result, LLM outputs, chatbot
      instance, etc.).
    - Scattering raw session_state access across ui/*.py pages makes
      keys easy to mistype and refactoring painful. This module owns
      every key name as a typed constant and exposes typed getter /
      setter helpers so pages never touch session_state directly.
    - Initialization is idempotent: call init_session_state() at the
      top of app.py and it only sets keys that don't already exist,
      so in-progress data is never wiped on re-run.

Key layout (all prefixed by their domain):
    resume.*    — upload, raw text, parsed/structured dicts
    jd.*        — job description text
    ats.*       — ATS score result dict
    match.*     — JD match result dict
    gap.*       — keyword gap analysis result dict
    review.*    — resume improvement suggestions dict
    interview.* — interview questions dict
    roadmap.*   — career roadmap dict
    chat.*      — chatbot instance + history
    nav.*       — current page for sidebar navigation

Usage:
    from utils.session_state import init_session_state, get, set_resume_text

    # In app.py:
    init_session_state()

    # In any ui/ page:
    from utils.session_state import (
        get_resume_text, set_resume_text,
        get_parsed_resume, set_parsed_resume,
        get_job_description, set_job_description,
        ...
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    # Only imported for type annotations; avoids importing heavy
    # dependencies at session_state module load time.
    from core.rag_chatbot import ResumeChatbot


# ---------------------------------------------------------------------------
# 1. KEY REGISTRY
#    Every session_state key used anywhere in the app is defined here.
#    Use these constants everywhere instead of string literals.
# ---------------------------------------------------------------------------

# -- Resume upload & parsing -----------------------------------------------
KEY_UPLOADED_FILE_NAME: str = "resume.uploaded_file_name"
KEY_RESUME_TEXT: str = "resume.raw_text"
KEY_PARSED_RESUME: str = "resume.parsed"          # pdf_parser schema (flat)
KEY_STRUCTURED_RESUME: str = "resume.structured"  # resume_structurer schema (nested)

# -- Job description --------------------------------------------------------
KEY_JOB_DESCRIPTION: str = "jd.text"

# -- ATS score --------------------------------------------------------------
KEY_ATS_RESULT: str = "ats.result"

# -- JD match ---------------------------------------------------------------
KEY_MATCH_RESULT: str = "match.result"

# -- Keyword gap ------------------------------------------------------------
KEY_GAP_RESULT: str = "gap.result"

# -- Resume suggestions -----------------------------------------------------
KEY_REVIEW_RESULT: str = "review.result"

# -- Interview questions ----------------------------------------------------
KEY_INTERVIEW_RESULT: str = "interview.result"

# -- Career roadmap ---------------------------------------------------------
KEY_ROADMAP_RESULT: str = "roadmap.result"

# -- RAG chatbot ------------------------------------------------------------
KEY_CHATBOT: str = "chat.bot"                  # ResumeChatbot instance
KEY_CHAT_MESSAGES: str = "chat.messages"       # UI message log (display only)

# -- Navigation -------------------------------------------------------------
KEY_CURRENT_PAGE: str = "nav.current_page"


# ---------------------------------------------------------------------------
# 2. DEFAULT VALUES
#    Defines the initial/reset value for every key.
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    KEY_UPLOADED_FILE_NAME: None,
    KEY_RESUME_TEXT:        "",
    KEY_PARSED_RESUME:      None,
    KEY_STRUCTURED_RESUME:  None,
    KEY_JOB_DESCRIPTION:    "",
    KEY_ATS_RESULT:         None,
    KEY_MATCH_RESULT:       None,
    KEY_GAP_RESULT:         None,
    KEY_REVIEW_RESULT:      None,
    KEY_INTERVIEW_RESULT:   None,
    KEY_ROADMAP_RESULT:     None,
    KEY_CHATBOT:            None,
    KEY_CHAT_MESSAGES:      [],
    KEY_CURRENT_PAGE:       None,
}


# ---------------------------------------------------------------------------
# 3. INITIALIZATION
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """
    Idempotently initialize all session_state keys to their defaults.

    Call this exactly once at the top of app.py, before any page is
    rendered. Keys that already exist (because the user has already
    uploaded a resume, etc.) are left untouched.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            # Lists must be initialized to new instances to avoid all
            # keys sharing the same mutable default object.
            if isinstance(default, list):
                st.session_state[key] = []
            else:
                st.session_state[key] = default


def reset_session_state(*, keep_resume: bool = False) -> None:
    """
    Reset all session_state keys to their default values.

    Useful for implementing a "Start Over" / "Clear All" button in
    the UI.

    Args:
        keep_resume: If True, preserve the uploaded file name, raw
                     text, parsed resume, and structured resume so the
                     user doesn't have to re-upload their file after
                     clearing LLM results. If False (default), reset
                     everything including the uploaded resume.
    """
    resume_keys = {
        KEY_UPLOADED_FILE_NAME,
        KEY_RESUME_TEXT,
        KEY_PARSED_RESUME,
        KEY_STRUCTURED_RESUME,
    }

    for key, default in _DEFAULTS.items():
        if keep_resume and key in resume_keys:
            continue

        if isinstance(default, list):
            st.session_state[key] = []
        else:
            st.session_state[key] = default


def reset_llm_results() -> None:
    """
    Clear only the LLM-generated result caches (ATS, match, gap,
    review, interview, roadmap, chatbot).

    Call this whenever a new resume is uploaded so that stale results
    from the previous resume are not displayed alongside the new one.
    The raw resume text and parsed structures are kept intact.
    """
    llm_keys = [
        KEY_ATS_RESULT,
        KEY_MATCH_RESULT,
        KEY_GAP_RESULT,
        KEY_REVIEW_RESULT,
        KEY_INTERVIEW_RESULT,
        KEY_ROADMAP_RESULT,
        KEY_CHATBOT,
        KEY_CHAT_MESSAGES,
    ]
    for key in llm_keys:
        default = _DEFAULTS[key]
        st.session_state[key] = [] if isinstance(default, list) else default


# ---------------------------------------------------------------------------
# 4. GENERIC GET / SET
# ---------------------------------------------------------------------------

def get(key: str, default: Any = None) -> Any:
    """
    Retrieve a value from session_state by key.

    Args:
        key: The session_state key (use the KEY_* constants above).
        default: Value to return if the key is not present.

    Returns:
        The stored value, or `default` if the key is missing.
    """
    return st.session_state.get(key, default)


def set_value(key: str, value: Any) -> None:
    """
    Store a value in session_state.

    Args:
        key: The session_state key (use the KEY_* constants above).
        value: The value to store.
    """
    st.session_state[key] = value


def has_value(key: str) -> bool:
    """
    Check whether a key in session_state holds a non-None, non-empty
    value.

    Args:
        key: The session_state key to inspect.

    Returns:
        True if the key exists and its value is truthy (non-None,
        non-empty string, non-empty list, non-zero, etc.).
        False if the key is missing or its value is falsy.
    """
    value = st.session_state.get(key)
    return value is not None and value != "" and value != []


# ---------------------------------------------------------------------------
# 5. TYPED RESUME ACCESSORS
# ---------------------------------------------------------------------------

def get_resume_text() -> str:
    """Return the raw extracted resume text, or "" if not yet uploaded."""
    return st.session_state.get(KEY_RESUME_TEXT, "")


def set_resume_text(text: str) -> None:
    """Store the raw extracted resume text."""
    st.session_state[KEY_RESUME_TEXT] = text


def get_parsed_resume() -> Optional[Dict]:
    """
    Return the flat parsed resume dict (pdf_parser schema) or None.

    Schema (from core/pdf_parser.py):
        {"name": str, "email": str, "phone": str,
         "skills": List[str], "education": List[str],
         "experience": List[str], "projects": List[str]}
    """
    return st.session_state.get(KEY_PARSED_RESUME)


def set_parsed_resume(parsed: Dict) -> None:
    """
    Store the flat parsed resume (pdf_parser output).

    Also clears all downstream LLM results since a new resume
    invalidates any previously computed scores.
    """
    st.session_state[KEY_PARSED_RESUME] = parsed
    reset_llm_results()


def get_structured_resume() -> Optional[Dict]:
    """
    Return the structured resume dict (resume_structurer schema) or None.

    Schema (from core/resume_structurer.py):
        {"contact_info": {"name": str, "email": str, "phone": str},
         "skills": List[str], "education": List[str],
         "experience": List[str], "projects": List[str],
         "certifications": List[str]}
    """
    return st.session_state.get(KEY_STRUCTURED_RESUME)


def set_structured_resume(structured: Dict) -> None:
    """Store the structured resume (resume_structurer output)."""
    st.session_state[KEY_STRUCTURED_RESUME] = structured


def get_uploaded_file_name() -> Optional[str]:
    """Return the filename of the most recently uploaded PDF."""
    return st.session_state.get(KEY_UPLOADED_FILE_NAME)


def set_uploaded_file_name(name: str) -> None:
    """Store the filename of the uploaded PDF."""
    st.session_state[KEY_UPLOADED_FILE_NAME] = name


def resume_is_uploaded() -> bool:
    """
    Check whether a resume has been successfully uploaded and parsed.

    Returns True only when both the raw text and the parsed dict are
    present and non-empty.
    """
    text = get_resume_text()
    parsed = get_parsed_resume()
    return bool(text and text.strip() and parsed is not None)


# ---------------------------------------------------------------------------
# 6. JOB DESCRIPTION ACCESSORS
# ---------------------------------------------------------------------------

def get_job_description() -> str:
    """Return the current job description text, or ""."""
    return st.session_state.get(KEY_JOB_DESCRIPTION, "")


def set_job_description(text: str) -> None:
    """
    Store the job description text.

    Also clears match/gap results since a new JD invalidates them.
    """
    st.session_state[KEY_JOB_DESCRIPTION] = text

    # Invalidate results that depend on the JD.
    st.session_state[KEY_MATCH_RESULT] = None
    st.session_state[KEY_GAP_RESULT] = None
    st.session_state[KEY_ATS_RESULT] = None


def jd_is_provided() -> bool:
    """Return True if a non-empty job description has been entered."""
    jd = get_job_description()
    return bool(jd and jd.strip())


# ---------------------------------------------------------------------------
# 7. ATS RESULT ACCESSORS
# ---------------------------------------------------------------------------

def get_ats_result() -> Optional[Dict]:
    """
    Return the ATS score result dict or None.

    Schema (from core/ats_scorer.py):
        {"overall_score": int,
         "skills_score": float, "experience_score": float,
         "projects_score": float, "education_score": float,
         "structure_score": float,
         "missing_keywords": List[str]}
    """
    return st.session_state.get(KEY_ATS_RESULT)


def set_ats_result(result: Dict) -> None:
    """Store the ATS score result dict."""
    st.session_state[KEY_ATS_RESULT] = result


# ---------------------------------------------------------------------------
# 8. JD MATCH RESULT ACCESSORS
# ---------------------------------------------------------------------------

def get_match_result() -> Optional[Dict]:
    """
    Return the JD match result dict or None.

    Schema (from core/jd_matcher.py):
        {"match_score": float,
         "matched_skills": List[str],
         "missing_skills": List[str]}
    """
    return st.session_state.get(KEY_MATCH_RESULT)


def set_match_result(result: Dict) -> None:
    """Store the JD match result dict."""
    st.session_state[KEY_MATCH_RESULT] = result


# ---------------------------------------------------------------------------
# 9. KEYWORD GAP RESULT ACCESSORS
# ---------------------------------------------------------------------------

def get_gap_result() -> Optional[Dict]:
    """
    Return the keyword gap analysis result dict or None.

    Schema (from core/keyword_gap.py):
        {"present_keywords": List[str],
         "missing_keywords": List[str],
         "total_keywords": int,
         "match_percentage": float}
    """
    return st.session_state.get(KEY_GAP_RESULT)


def set_gap_result(result: Dict) -> None:
    """Store the keyword gap analysis result dict."""
    st.session_state[KEY_GAP_RESULT] = result


# ---------------------------------------------------------------------------
# 10. REVIEW / SUGGESTIONS ACCESSORS
# ---------------------------------------------------------------------------

def get_review_result() -> Optional[Dict]:
    """
    Return the resume review result dict or None.

    Schema (from core/suggestion_engine.py):
        {"strengths": List[str],
         "weaknesses": List[str],
         "missing_sections": List[str],
         "improvement_suggestions": List[str]}
    """
    return st.session_state.get(KEY_REVIEW_RESULT)


def set_review_result(result: Dict) -> None:
    """Store the resume review result dict."""
    st.session_state[KEY_REVIEW_RESULT] = result


# ---------------------------------------------------------------------------
# 11. INTERVIEW QUESTIONS ACCESSORS
# ---------------------------------------------------------------------------

def get_interview_result() -> Optional[Dict]:
    """
    Return the interview questions result dict or None.

    Schema (from core/interview_generator.py):
        {"technical": List[str],
         "project": List[str],
         "behavioral": List[str]}
    """
    return st.session_state.get(KEY_INTERVIEW_RESULT)


def set_interview_result(result: Dict) -> None:
    """Store the interview questions result dict."""
    st.session_state[KEY_INTERVIEW_RESULT] = result


# ---------------------------------------------------------------------------
# 12. CAREER ROADMAP ACCESSORS
# ---------------------------------------------------------------------------

def get_roadmap_result() -> Optional[Dict]:
    """
    Return the career roadmap result dict or None.

    Schema (from core/career_roadmap.py):
        {"skill_gaps": List[str],
         "recommended_technologies": List[str],
         "learning_roadmap": List[Dict],   # {stage, duration, topics}
         "suggested_projects": List[str]}
    """
    return st.session_state.get(KEY_ROADMAP_RESULT)


def set_roadmap_result(result: Dict) -> None:
    """Store the career roadmap result dict."""
    st.session_state[KEY_ROADMAP_RESULT] = result


# ---------------------------------------------------------------------------
# 13. CHATBOT ACCESSORS
# ---------------------------------------------------------------------------

def get_chatbot() -> Optional["ResumeChatbot"]:
    """
    Return the ResumeChatbot instance or None.

    The chatbot is lazily constructed and stored here on first use
    by the chatbot page (ui/chatbot_page.py). Storing it in session
    state means the FAISS index built from the resume is preserved
    across Streamlit re-runs without re-indexing every time.
    """
    return st.session_state.get(KEY_CHATBOT)


def set_chatbot(bot: "ResumeChatbot") -> None:
    """Store the ResumeChatbot instance."""
    st.session_state[KEY_CHATBOT] = bot


def chatbot_is_ready() -> bool:
    """
    Return True if the chatbot exists and has an ingested resume.

    Calls ResumeChatbot.is_ready() to check that the FAISS index is
    non-empty (i.e. the resume has been chunked and indexed).
    """
    bot = get_chatbot()
    return bot is not None and bot.is_ready()


def get_chat_messages() -> List[Dict[str, str]]:
    """
    Return the UI chat message log (list of {role, content} dicts).

    This is the display-only log used to render the conversation in
    st.chat_message() calls. It mirrors the chatbot's internal history
    but is kept separately so the UI can render it independently of
    the chatbot's rolling window.

    Returns:
        List of {"role": "user"|"assistant", "content": str} dicts.
        Returns [] if no messages have been logged yet.
    """
    return st.session_state.get(KEY_CHAT_MESSAGES, [])


def append_chat_message(role: str, content: str) -> None:
    """
    Append a single message to the UI chat log.

    Args:
        role: "user" or "assistant".
        content: The message text.
    """
    messages = st.session_state.get(KEY_CHAT_MESSAGES, [])
    messages.append({"role": role, "content": content})
    st.session_state[KEY_CHAT_MESSAGES] = messages


def clear_chat_messages() -> None:
    """Clear the UI chat message log (does not affect the chatbot's internal history)."""
    st.session_state[KEY_CHAT_MESSAGES] = []


# ---------------------------------------------------------------------------
# 14. NAVIGATION ACCESSORS
# ---------------------------------------------------------------------------

def get_current_page() -> Optional[str]:
    """Return the currently active page name, or None on first load."""
    return st.session_state.get(KEY_CURRENT_PAGE)


def set_current_page(page_name: str) -> None:
    """Set the currently active page name."""
    st.session_state[KEY_CURRENT_PAGE] = page_name


# ---------------------------------------------------------------------------
# 15. CONVENIENCE: GUARD HELPERS FOR UI PAGES
# ---------------------------------------------------------------------------

def require_resume(page_name: str = "this page") -> bool:
    """
    Render an informational warning and return False if no resume has
    been uploaded yet.

    Use at the top of every ui/ page that needs resume data:

        if not require_resume("ATS Score"):
            return   # Stop rendering; the warning is already shown.

    Args:
        page_name: Human-readable name of the current page, shown in
                    the warning message.

    Returns:
        True if a resume is available (page can proceed).
        False if no resume is present (warning has been shown).
    """
    if resume_is_uploaded():
        return True

    st.warning(
        f"**{page_name}** needs a resume. "
        "Please go to **📄 Upload Resume** and upload your PDF first.",
        icon="📎",
    )
    return False


def require_job_description(page_name: str = "this page") -> bool:
    """
    Render an informational warning and return False if no job
    description has been entered yet.

    Args:
        page_name: Human-readable name of the current page.

    Returns:
        True if a JD is present, False otherwise (warning shown).
    """
    if jd_is_provided():
        return True

    st.warning(
        f"**{page_name}** needs a job description. "
        "Please enter one in the sidebar or on the JD Match page.",
        icon="📋",
    )
    return False


def get_debug_snapshot() -> Dict[str, Any]:
    """
    Return a snapshot of all managed session_state keys and their
    current values (or a summary for large objects).

    Intended for a developer-mode debug expander in the sidebar, not
    for production display.

    Returns:
        Dict mapping each KEY_* constant to a human-readable summary.
    """
    snapshot: Dict[str, Any] = {}

    for key in _DEFAULTS:
        value = st.session_state.get(key, "<missing>")

        if isinstance(value, str):
            summary = f"{len(value)} chars" if value else "(empty)"
        elif isinstance(value, list):
            summary = f"List[{len(value)} items]"
        elif isinstance(value, dict):
            summary = f"Dict[{list(value.keys())}]"
        elif value is None:
            summary = "None"
        else:
            # ResumeChatbot or other object.
            summary = type(value).__name__

        snapshot[key] = summary

    return snapshot
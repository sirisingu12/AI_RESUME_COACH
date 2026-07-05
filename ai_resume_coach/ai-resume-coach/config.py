"""
config.py
---------
Central configuration for the AI Resume Analyzer & Career Coach.

All paths, endpoint URLs, model names, thresholds, and UI constants
are defined here. Every other module (core/, ui/, utils/) should
import from this file rather than hardcoding values locally.

Usage:
    from config import LM_STUDIO_BASE_URL, EMBEDDING_MODEL, FAISS_INDEX_PATH
    from config import ATS_WEIGHTS, AppConfig

Environment override:
    Any value can be overridden at startup by setting the matching
    environment variable (documented per constant below). This lets
    you run against a remote LM Studio instance or a different model
    without touching source code.

    Example:
        LM_STUDIO_URL=http://192.168.1.10:1234/v1 streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# 1. BASE PATHS
# ---------------------------------------------------------------------------

# Root of the project (the directory this file lives in).
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Sub-directories — created at runtime by the app if missing.
DATA_DIR: Path = PROJECT_ROOT / "data"
SAMPLE_RESUMES_DIR: Path = DATA_DIR / "sample_resumes"
SAMPLE_JD_DIR: Path = DATA_DIR / "sample_job_descriptions"
FAISS_INDEX_DIR: Path = DATA_DIR / "faiss_index"

# ---------------------------------------------------------------------------
# 2. FAISS / VECTOR STORE
# ---------------------------------------------------------------------------

# Base path (no extension) passed to VectorStore.save() / VectorStore.load().
# VectorStore appends ".index" and ".meta.json" automatically.
FAISS_INDEX_PATH: str = str(FAISS_INDEX_DIR / "resume_index")

# Maximum characters per resume chunk when building the RAG index.
# Mirrors rag_chatbot.MAX_CHUNK_CHARS; defined here so it can be
# tweaked in one place.
RAG_CHUNK_MAX_CHARS: int = 500

# Number of chunks retrieved per RAG query.
RAG_TOP_K: int = 5

# Maximum number of prior conversation turns kept in the chatbot's
# rolling history (each turn = 1 user message + 1 assistant message,
# so MAX_HISTORY_TURNS=5 keeps 10 messages total).
RAG_MAX_HISTORY_TURNS: int = 5

# ---------------------------------------------------------------------------
# 3. LM STUDIO / LLM
# ---------------------------------------------------------------------------

# Base URL of the LM Studio local server.
# Override with the LM_STUDIO_URL environment variable.
LM_STUDIO_BASE_URL: str = os.environ.get(
    "LM_STUDIO_URL", "http://localhost:1234/v1"
)

# Model identifier sent in every /chat/completions request.
# "local-model" is accepted by most LM Studio versions as a generic
# alias, but setting this to the exact model name shown in LM Studio's
# "Local Server" tab is recommended.
# Override with the LM_STUDIO_MODEL environment variable.
LM_STUDIO_MODEL: str = os.environ.get("LM_STUDIO_MODEL", "llama-3.2-3b-instruct")

# Request timeout in seconds. Increase for slower hardware or large
# models that take a long time to generate.
LLM_TIMEOUT_SECONDS: int = int(os.environ.get("LLM_TIMEOUT", "300"))

# Sampling temperature for all LLM calls (0.0 = deterministic, 1.0 = creative).
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.7"))

# Default max tokens generated per LLM call. Individual task functions
# override this where appropriate.
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

# Per-task token budgets (passed as max_tokens overrides in each module).
LLM_MAX_TOKENS_REVIEW: int = 1500
LLM_MAX_TOKENS_INTERVIEW: int = 1800
LLM_MAX_TOKENS_ROADMAP: int = 2000
LLM_MAX_TOKENS_CHAT: int = 800

# ---------------------------------------------------------------------------
# 4. EMBEDDING MODEL
# ---------------------------------------------------------------------------

# Name of the sentence-transformers model used for all embeddings.
# Must match the value in embeddings.py (MODEL_NAME).
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Cosine similarity threshold used in jd_matcher for semantic skill matching.
# JD keywords with similarity >= this threshold against any resume keyword
# are considered matched even without an exact text match.
SEMANTIC_MATCH_THRESHOLD: float = 0.55

# ---------------------------------------------------------------------------
# 5. ATS SCORING
# ---------------------------------------------------------------------------

# Point allocation for each resume section in the ATS score.
# Must sum to 100. Mirrors ats_scorer.WEIGHTS.
ATS_WEIGHTS: Dict[str, int] = {
    "skills": 40,
    "experience": 25,
    "projects": 15,
    "education": 10,
    "structure": 10,
}

# Score band labels displayed on the ATS results page.
ATS_SCORE_BANDS: Dict[str, tuple] = {
    "Excellent": (85, 100),
    "Good":      (70, 84),
    "Fair":      (50, 69),
    "Weak":      (0,  49),
}

# ---------------------------------------------------------------------------
# 6. INTERVIEW GENERATOR
# ---------------------------------------------------------------------------

# Default number of questions requested per category (technical /
# project / behavioral).
INTERVIEW_QUESTIONS_PER_CATEGORY: int = 5

# ---------------------------------------------------------------------------
# 7. PDF PARSING
# ---------------------------------------------------------------------------

# Maximum file size (in bytes) accepted for PDF upload. 10 MB default.
MAX_PDF_SIZE_BYTES: int = int(os.environ.get("MAX_PDF_MB", "10")) * 1024 * 1024

# Accepted MIME types for the Streamlit file uploader.
ACCEPTED_UPLOAD_TYPES: list = ["pdf"]

# ---------------------------------------------------------------------------
# 8. UI / STREAMLIT
# ---------------------------------------------------------------------------

# Application title shown in the browser tab and sidebar.
APP_TITLE: str = "AI Resume Coach"

# Subtitle shown under the main heading.
APP_SUBTITLE: str = "Analyze, optimize, and coach your career — powered by local AI"

# Streamlit page layout ("centered" or "wide").
PAGE_LAYOUT: str = "wide"

# Sidebar state on first load ("expanded" or "collapsed").
INITIAL_SIDEBAR_STATE: str = "expanded"

# Navigation page names — must match exactly what ui/ pages register.
PAGE_UPLOAD: str = "📄 Upload Resume"
PAGE_ATS: str = "📊 ATS Score"
PAGE_JD_MATCH: str = "🎯 JD Match"
PAGE_SUGGESTIONS: str = "💡 Suggestions"
PAGE_INTERVIEW: str = "🎤 Interview Prep"
PAGE_ROADMAP: str = "🗺️ Career Roadmap"
PAGE_CHATBOT: str = "💬 Resume Chatbot"

# Ordered list used to build the sidebar nav (order matters).
NAV_PAGES: list = [
    PAGE_UPLOAD,
    PAGE_ATS,
    PAGE_JD_MATCH,
    PAGE_SUGGESTIONS,
    PAGE_INTERVIEW,
    PAGE_ROADMAP,
    PAGE_CHATBOT,
]

# Pages that require a resume to be uploaded before they can be used.
PAGES_REQUIRING_RESUME: set = {
    PAGE_ATS,
    PAGE_JD_MATCH,
    PAGE_SUGGESTIONS,
    PAGE_INTERVIEW,
    PAGE_ROADMAP,
    PAGE_CHATBOT,
}

# Pages that additionally require a job description to be entered.
PAGES_REQUIRING_JD: set = {
    PAGE_JD_MATCH,
}

# ---------------------------------------------------------------------------
# 9. PLOTLY / CHART COLOURS
# ---------------------------------------------------------------------------

# Primary accent colour used on charts and call-outs (matches the
# Streamlit theme primary in .streamlit/config.toml if you set one).
CHART_PRIMARY_COLOR: str = "#4F8BF9"

# Colour sequence for multi-series charts (skills breakdown bar chart etc.)
CHART_COLOR_SEQUENCE: list = [
    "#4F8BF9",  # blue
    "#43C59E",  # teal
    "#F97B4F",  # orange
    "#A855F7",  # purple
    "#F4C542",  # amber
]

# Score band colours used on the ATS gauge.
ATS_BAND_COLORS: Dict[str, str] = {
    "Excellent": "#43C59E",   # green
    "Good":      "#4F8BF9",   # blue
    "Fair":      "#F4C542",   # amber
    "Weak":      "#F97B4F",   # red-orange
}

# ---------------------------------------------------------------------------
# 10. CONVENIENCE: dataclass-style namespace for imports
# ---------------------------------------------------------------------------

class AppConfig:
    """
    Namespace class that exposes every constant above as a class
    attribute. Useful when you want to pass the entire configuration
    around as a single object rather than importing individual names.

    Usage:
        from config import AppConfig
        url = AppConfig.LM_STUDIO_BASE_URL
    """

    # Paths
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = DATA_DIR
    FAISS_INDEX_PATH = FAISS_INDEX_PATH

    # LLM
    LM_STUDIO_BASE_URL = LM_STUDIO_BASE_URL
    LM_STUDIO_MODEL = LM_STUDIO_MODEL
    LLM_TIMEOUT_SECONDS = LLM_TIMEOUT_SECONDS
    LLM_TEMPERATURE = LLM_TEMPERATURE
    LLM_MAX_TOKENS = LLM_MAX_TOKENS
    LLM_MAX_TOKENS_REVIEW = LLM_MAX_TOKENS_REVIEW
    LLM_MAX_TOKENS_INTERVIEW = LLM_MAX_TOKENS_INTERVIEW
    LLM_MAX_TOKENS_ROADMAP = LLM_MAX_TOKENS_ROADMAP
    LLM_MAX_TOKENS_CHAT = LLM_MAX_TOKENS_CHAT

    # Embeddings / RAG
    EMBEDDING_MODEL = EMBEDDING_MODEL
    SEMANTIC_MATCH_THRESHOLD = SEMANTIC_MATCH_THRESHOLD
    RAG_CHUNK_MAX_CHARS = RAG_CHUNK_MAX_CHARS
    RAG_TOP_K = RAG_TOP_K
    RAG_MAX_HISTORY_TURNS = RAG_MAX_HISTORY_TURNS

    # ATS
    ATS_WEIGHTS = ATS_WEIGHTS
    ATS_SCORE_BANDS = ATS_SCORE_BANDS

    # Interview
    INTERVIEW_QUESTIONS_PER_CATEGORY = INTERVIEW_QUESTIONS_PER_CATEGORY

    # Upload
    MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_BYTES
    ACCEPTED_UPLOAD_TYPES = ACCEPTED_UPLOAD_TYPES

    # UI
    APP_TITLE = APP_TITLE
    APP_SUBTITLE = APP_SUBTITLE
    PAGE_LAYOUT = PAGE_LAYOUT
    INITIAL_SIDEBAR_STATE = INITIAL_SIDEBAR_STATE
    NAV_PAGES = NAV_PAGES
    PAGES_REQUIRING_RESUME = PAGES_REQUIRING_RESUME
    PAGES_REQUIRING_JD = PAGES_REQUIRING_JD

    # Charts
    CHART_PRIMARY_COLOR = CHART_PRIMARY_COLOR
    CHART_COLOR_SEQUENCE = CHART_COLOR_SEQUENCE
    ATS_BAND_COLORS = ATS_BAND_COLORS


# ---------------------------------------------------------------------------
# 11. RUNTIME DIRECTORY BOOTSTRAP
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """
    Create all required data directories if they do not already exist.

    Call this once at app startup (in app.py) so that the first run
    on a clean machine does not crash when VectorStore tries to write
    an index or pdf_parser tries to access sample data.
    """
    for directory in (DATA_DIR, SAMPLE_RESUMES_DIR, SAMPLE_JD_DIR, FAISS_INDEX_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Manual test / sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ensure_directories()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"FAISS index  : {FAISS_INDEX_PATH}")
    print(f"LM Studio URL: {LM_STUDIO_BASE_URL}")
    print(f"Model        : {LM_STUDIO_MODEL}")
    print(f"ATS weights  : {ATS_WEIGHTS}  (sum={sum(ATS_WEIGHTS.values())})")
    print("All data directories ensured.")
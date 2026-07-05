"""
core
----
Core processing package for the AI Resume Analyzer & Career Coach.

This package contains all non-UI logic: PDF parsing, resume
structuring, embeddings, FAISS vector storage, ATS scoring, JD
matching, keyword gap analysis, and the RAG-based resume chatbot.

Re-exports the most commonly used classes/functions so callers can do:

    from core import structure_resume, calculate_ats_score, ResumeChatbot

instead of reaching into each submodule individually.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Resume parsing & structuring
# ---------------------------------------------------------------------------
from .pdf_parser import extract_text_from_pdf
from .resume_structurer import structure_resume, clean_text

# ---------------------------------------------------------------------------
# Embeddings & vector store
# ---------------------------------------------------------------------------
from embeddings import (
    create_embedding,
    create_embeddings_batch,
    get_model as get_embedding_model,
    EmbeddingError,
)
from .vector_store import VectorStore, VectorStoreError

# ---------------------------------------------------------------------------
# Scoring & matching
# ---------------------------------------------------------------------------
from .ats_scorer import calculate_ats_score
from .jd_matcher import match_resume_to_job
from .keyword_gap import analyze_keyword_gap

# ---------------------------------------------------------------------------
# LLM-powered features
# ---------------------------------------------------------------------------
from .suggestion_engine import review_resume
from .interview_generator import generate_interview_questions
from .career_roadmap import generate_roadmap
from .rag_chatbot import ResumeChatbot, chunk_resume_text


__all__ = [
    # Parsing & structuring
    "extract_text_from_pdf",
    "structure_resume",
    "clean_text",
    # Embeddings & vector store
    "create_embedding",
    "create_embeddings_batch",
    "get_embedding_model",
    "EmbeddingError",
    "VectorStore",
    "VectorStoreError",
    # Scoring & matching
    "calculate_ats_score",
    "match_resume_to_job",
    "analyze_keyword_gap",
    # LLM-powered features
    "review_resume",
    "generate_interview_questions",
    "generate_roadmap",
    "ResumeChatbot",
    "chunk_resume_text",
]

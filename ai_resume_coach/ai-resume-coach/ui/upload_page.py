"""
ui/upload_page.py
------------------
Resume upload page.

Responsibilities:
    - Accept a PDF resume upload (via st.file_uploader).
    - Validate file size against config.MAX_PDF_SIZE_BYTES.
    - Extract raw text via core.pdf_parser.extract_text_from_pdf().
    - Structure the resume via core.resume_structurer.structure_resume()
      (nested schema) for downstream consumers, and additionally store
      a flat schema for backward-compatible consumers that expect the
      pdf_parser-style dict.
    - Display a preview of the extracted/structured data so the user
      can confirm the upload worked before moving to other pages.

This page is the entry point of the app: resume_is_uploaded() in
session_state becomes True only after a successful upload here.
"""

from __future__ import annotations

import streamlit as st

from config import MAX_PDF_SIZE_BYTES, ACCEPTED_UPLOAD_TYPES
from core.pdf_parser import extract_text_from_pdf
from core.resume_structurer import structure_resume
from utils.session_state import (
    set_resume_text,
    set_parsed_resume,
    set_structured_resume,
    set_uploaded_file_name,
    get_resume_text,
    get_structured_resume,
    get_uploaded_file_name,
    resume_is_uploaded,
)
from utils.text_cleaning import count_words, is_meaningful_text, truncate_for_display


def _structured_to_flat(structured: dict) -> dict:
    """
    Convert the nested resume_structurer schema into the flat
    pdf_parser-style schema, for consumers (e.g. ats_scorer) that
    expect top-level "name"/"email"/"phone" keys.

    Args:
        structured: The dict returned by core.resume_structurer.structure_resume().

    Returns:
        A flat dict with keys: "name", "email", "phone", "skills",
        "education", "projects", "experience". Note: "certifications"
        from the structured schema is not part of the flat schema and
        is intentionally dropped here.
    """
    contact = structured.get("contact_info", {}) or {}
    return {
        "name": contact.get("name", ""),
        "email": contact.get("email", ""),
        "phone": contact.get("phone", ""),
        "skills": structured.get("skills", []),
        "education": structured.get("education", []),
        "projects": structured.get("projects", []),
        "experience": structured.get("experience", []),
    }


def _validate_upload(uploaded_file) -> str | None:
    """
    Validate an uploaded file against size and type constraints.

    Args:
        uploaded_file: The object returned by st.file_uploader().

    Returns:
        An error message string if validation fails, or None if the
        file is acceptable.
    """
    if uploaded_file.size == 0:
        return "The uploaded file is empty."

    if uploaded_file.size > MAX_PDF_SIZE_BYTES:
        max_mb = MAX_PDF_SIZE_BYTES / (1024 * 1024)
        actual_mb = uploaded_file.size / (1024 * 1024)
        return f"File is too large ({actual_mb:.1f} MB). Maximum allowed size is {max_mb:.0f} MB."

    return None


def _process_upload(uploaded_file) -> None:
    """
    Extract, structure, and store a newly uploaded resume PDF.

    On success, updates session_state (raw text, flat parsed dict,
    structured dict, uploaded filename) and clears any stale LLM
    results from a previous resume. On failure, shows an error and
    leaves session_state unchanged.

    Args:
        uploaded_file: The object returned by st.file_uploader().
    """
    validation_error = _validate_upload(uploaded_file)
    if validation_error:
        st.error(validation_error, icon="⚠️")
        return

    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    if not raw_text or not raw_text.strip():
        st.error(
            "Could not extract any text from this PDF. It may be a "
            "scanned/image-only document, password-protected, or "
            "corrupted. Please try a different file.",
            icon="⚠️",
        )
        return

    if not is_meaningful_text(raw_text, min_words=20):
        st.warning(
            "Only a small amount of text was extracted from this PDF. "
            "Results on other pages may be limited. Continuing anyway.",
            icon="⚠️",
        )

    with st.spinner("Structuring resume content..."):
        structured = structure_resume(raw_text)

    flat_parsed = _structured_to_flat(structured)

    # set_parsed_resume() also clears stale ATS/match/gap/review/
    # interview/roadmap/chatbot results from any previously uploaded
    # resume.
    set_resume_text(raw_text)
    set_parsed_resume(flat_parsed)
    set_structured_resume(structured)
    set_uploaded_file_name(uploaded_file.name)

    st.success(f"Successfully processed **{uploaded_file.name}**.", icon="✅")


def _render_preview() -> None:
    """
    Render a read-only preview of the currently stored resume: raw
    text stats, contact info, and extracted sections.
    """
    structured = get_structured_resume()
    if structured is None:
        return

    raw_text = get_resume_text()
    contact = structured.get("contact_info", {}) or {}

    st.subheader("Preview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Words extracted", count_words(raw_text))
    col2.metric("Skills found", len(structured.get("skills", [])))
    col3.metric("Experience entries", len(structured.get("experience", [])))

    with st.expander("Contact information", expanded=True):
        st.write(f"**Name:** {contact.get('name') or '_Not detected_'}")
        st.write(f"**Email:** {contact.get('email') or '_Not detected_'}")
        st.write(f"**Phone:** {contact.get('phone') or '_Not detected_'}")

    section_labels = {
        "skills": "Skills",
        "education": "Education",
        "experience": "Experience",
        "projects": "Projects",
        "certifications": "Certifications",
    }

    for key, label in section_labels.items():
        items = structured.get(key, [])
        with st.expander(f"{label} ({len(items)})"):
            if items:
                for item in items:
                    st.markdown(f"- {item}")
            else:
                st.caption(f"No {label.lower()} detected.")

    with st.expander("Raw extracted text"):
        st.text(truncate_for_display(raw_text, max_chars=3000))


def render() -> None:
    """
    Render the Resume Upload page.

    Workflow:
        1. Show a file uploader restricted to PDF files.
        2. On a new file, validate, extract, structure, and store it.
        3. Always show a preview of the currently stored resume (if
           any), so re-visiting this page after upload still shows
           the result.
    """
    st.write(
        "Upload your resume as a PDF. It will be parsed locally — no "
        "data leaves your machine."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=ACCEPTED_UPLOAD_TYPES,
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        # Avoid re-processing the same file on every re-run: only
        # process if this filename differs from the one already stored.
        if uploaded_file.name != get_uploaded_file_name():
            _process_upload(uploaded_file)
        else:
            st.info(f"Using previously processed file **{uploaded_file.name}**.", icon="📄")

    st.divider()

    if resume_is_uploaded():
        _render_preview()
    else:
        st.caption("No resume uploaded yet. Choose a PDF file above to get started.")

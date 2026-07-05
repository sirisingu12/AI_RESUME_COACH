"""
ui/chatbot_page.py
---------------------
Resume Chatbot page (RAG).

Responsibilities:
    - Lazily create a core.rag_chatbot.ResumeChatbot instance and
      store it in session_state (chat.bot).
    - Ingest the current resume text into the chatbot's vector index
      (only when the resume changes, tracked via a hash in
      session_state).
    - Render a chat interface using st.chat_message() /
      st.chat_input(), backed by ResumeChatbot.ask().
    - Maintain a separate display-only message log (chat.messages) so
      the conversation renders correctly across re-runs.

Notes:
    - ResumeChatbot.ingest_resume() can raise ValueError /
      VectorStoreError on failure (e.g. embedding errors); these are
      caught and shown as st.error() rather than crashing the page.
    - ResumeChatbot.ask() never raises — LLM/connection errors are
      returned as "[LLM Error] ..." strings, which are displayed as
      the assistant's message (st.chat_message handles plain text
      fine either way), but we additionally surface them via
      st.error() for visibility.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from core.rag_chatbot import ResumeChatbot
from core.vector_store import VectorStoreError
from utils.session_state import (
    require_resume,
    get_resume_text,
    get_chatbot,
    set_chatbot,
    chatbot_is_ready,
    get_chat_messages,
    append_chat_message,
    clear_chat_messages,
    get,
    set_value,
)


# session_state key (local to this page) tracking which resume text
# hash the chatbot's index was last built from.
_KEY_INDEXED_RESUME_HASH = "chat.indexed_resume_hash"


def _resume_hash(resume_text: str) -> str:
    """
    Compute a short hash of the resume text, used to detect whether
    the chatbot's vector index needs to be rebuilt.

    Args:
        resume_text: The full raw resume text.

    Returns:
        A hex digest string uniquely identifying this resume text.
    """
    return hashlib.sha256(resume_text.encode("utf-8")).hexdigest()


def _ensure_chatbot_ready(resume_text: str) -> bool:
    """
    Ensure a ResumeChatbot exists in session_state and its vector
    index reflects the current resume text, (re)ingesting if needed.

    Args:
        resume_text: The full raw resume text.

    Returns:
        True if the chatbot is ready to answer questions, False if
        ingestion failed (an error has already been shown).
    """
    bot = get_chatbot()
    if bot is None:
        bot = ResumeChatbot()
        set_chatbot(bot)

    current_hash = _resume_hash(resume_text)
    indexed_hash = get(_KEY_INDEXED_RESUME_HASH)

    if indexed_hash == current_hash and bot.is_ready():
        return True

    try:
        with st.spinner("Indexing your resume for the chatbot..."):
            num_chunks = bot.ingest_resume(resume_text)
    except (ValueError, VectorStoreError) as exc:
        st.error(f"Failed to prepare the chatbot: {exc}", icon="⚠️")
        return False

    set_value(_KEY_INDEXED_RESUME_HASH, current_hash)
    clear_chat_messages()
    st.toast(f"Resume indexed into {num_chunks} chunk(s) for the chatbot.", icon="✅")
    return True


def _render_chat_history() -> None:
    """
    Render all messages in the display-only chat log via
    st.chat_message().
    """
    for message in get_chat_messages():
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


def render() -> None:
    """
    Render the Resume Chatbot page.

    Workflow:
        1. Ensure a resume is uploaded (require_resume guard).
        2. Lazily build/refresh the RAG vector index for the current
           resume.
        3. Render the existing conversation.
        4. Accept new questions via st.chat_input(), answer them via
           ResumeChatbot.ask(), and append both turns to the display
           log.
    """
    if not require_resume("Resume Chatbot"):
        return

    st.write(
        "Ask questions about your resume — e.g. \"What programming "
        "languages do I know?\" or \"Summarize my most recent "
        "experience.\" Answers are generated locally using your "
        "resume content as context."
    )

    resume_text = get_resume_text()

    if not _ensure_chatbot_ready(resume_text):
        return

    if not chatbot_is_ready():
        st.info("The chatbot is not ready yet. Try re-uploading your resume.", icon="ℹ️")
        return

    col1, _ = st.columns([1, 3])
    if col1.button("🗑️ Clear conversation"):
        clear_chat_messages()
        bot = get_chatbot()
        if bot is not None:
            bot.reset_conversation()
        st.rerun()

    st.divider()

    _render_chat_history()

    question = st.chat_input("Ask something about your resume...")

    if question:
        append_chat_message("user", question)
        with st.chat_message("user"):
            st.markdown(question)

        bot = get_chatbot()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = bot.ask(question)

            if answer.startswith("[LLM Error]"):
                st.error(answer, icon="⚠️")
            else:
                st.markdown(answer)

        append_chat_message("assistant", answer)

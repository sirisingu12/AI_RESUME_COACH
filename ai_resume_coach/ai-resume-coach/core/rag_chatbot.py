"""
core/rag_chatbot.py
--------------------
Module responsible for a Retrieval-Augmented Generation (RAG) chatbot
that answers questions about a candidate's resume.

Workflow:
    Resume Text
        -> Chunking
        -> Embeddings (via VectorStore / embeddings.py)
        -> FAISS index (via VectorStore)
        -> Retrieval (VectorStore.search)
        -> Context Injection (prompt construction)
        -> LM Studio (llm.chat_with_resume)
        -> Answer

Provides:
    - ResumeChatbot: a class that owns a VectorStore instance, ingests
      resume text by chunking + indexing it, and answers questions by
      retrieving relevant chunks and passing them to the local LLM as
      context.

Integrates with:
    - embeddings.py (indirectly, via VectorStore)
    - core/vector_store.py (VectorStore)
    - llm.py (chat_with_resume, LLMClient)

Design notes:
    - Chunking is paragraph/line-based with a maximum chunk size, so
      each resume section (skills, experience, etc.) tends to become
      one or a few chunks rather than being split mid-sentence.
    - The chatbot maintains a short rolling chat history so follow-up
      questions ("what about the second one?") retain some context.
    - All errors are handled gracefully and surfaced as plain strings
      (never raised to the caller from `ask()`), consistent with
      llm.py's "[LLM Error] ..." convention.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from core.vector_store import VectorStore, VectorStoreError
from llm import LLMClient, chat_with_resume
from config import RAG_CHUNK_MAX_CHARS, RAG_TOP_K, RAG_MAX_HISTORY_TURNS


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of characters per chunk. Sourced from config.py
# (RAG_CHUNK_MAX_CHARS) so it can be tuned in one place.
MAX_CHUNK_CHARS: int = RAG_CHUNK_MAX_CHARS

# Number of chunks retrieved per question. Sourced from config.py.
DEFAULT_TOP_K: int = RAG_TOP_K

# Maximum number of prior user/assistant turn pairs kept in chat
# history and passed to the LLM (older turns are dropped). Sourced
# from config.py.
MAX_HISTORY_TURNS: int = RAG_MAX_HISTORY_TURNS


# ---------------------------------------------------------------------------
# 1. CHUNKING
# ---------------------------------------------------------------------------
def chunk_resume_text(resume_text: str, max_chunk_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    Split resume text into retrieval-friendly chunks.

    Strategy:
        - Split on blank-line-separated paragraphs/sections first
          (preserves logical groupings like "Skills: ...",
          "Experience: ...").
        - Any resulting chunk longer than `max_chunk_chars` is further
          split on line boundaries, then on sentence boundaries if a
          single line is still too long.

    Args:
        resume_text: The full resume text to chunk (e.g. structured
                      sections joined together, or raw extracted text).
        max_chunk_chars: Maximum number of characters per chunk.

    Returns:
        A list of non-empty, stripped text chunks. Returns an empty
        list if `resume_text` is empty or whitespace-only.
    """
    if not resume_text or not resume_text.strip():
        return []

    # Split on one-or-more blank lines to get logical paragraphs/sections.
    raw_paragraphs = re.split(r"\n\s*\n", resume_text.strip())

    chunks: List[str] = []
    for paragraph in raw_paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) <= max_chunk_chars:
            chunks.append(paragraph)
        else:
            chunks.extend(_split_long_paragraph(paragraph, max_chunk_chars))

    logger.debug("chunk_resume_text produced %d chunks.", len(chunks))
    return chunks


def _split_long_paragraph(paragraph: str, max_chunk_chars: int) -> List[str]:
    """
    Split a single long paragraph into smaller chunks, first by line,
    then by sentence if individual lines are still too long.

    Args:
        paragraph: The paragraph text to split (already stripped,
                    longer than `max_chunk_chars`).
        max_chunk_chars: Maximum number of characters per chunk.

    Returns:
        A list of non-empty chunks, each at or below
        `max_chunk_chars` where reasonably possible (a single very
        long sentence with no break points may still exceed it).
    """
    chunks: List[str] = []
    current = ""

    for line in paragraph.splitlines():
        line = line.strip()
        if not line:
            continue

        pieces = [line] if len(line) <= max_chunk_chars else _split_by_sentence(line, max_chunk_chars)

        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= max_chunk_chars:
                current = f"{current} {piece}"
            else:
                chunks.append(current)
                current = piece

    if current:
        chunks.append(current)

    return chunks


def _split_by_sentence(line: str, max_chunk_chars: int) -> List[str]:
    """
    Split a single long line into sentence-sized pieces.

    Args:
        line: A single line of text longer than `max_chunk_chars`.
        max_chunk_chars: Maximum number of characters per piece.

    Returns:
        A list of sentence-level (or smaller) pieces.
    """
    sentences = re.split(r"(?<=[.!?])\s+", line)

    pieces: List[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chunk_chars:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    return pieces


# ---------------------------------------------------------------------------
# 2. ResumeChatbot
# ---------------------------------------------------------------------------
class ResumeChatbot:
    """
    A retrieval-augmented chatbot for answering questions about a
    candidate's resume.

    Owns a VectorStore instance and a rolling chat history. A resume
    must be ingested via `ingest_resume()` before `ask()` can return
    meaningful answers.

    Attributes:
        vector_store: The VectorStore used for chunk retrieval.
        top_k: Number of chunks retrieved per question.
        max_chunk_chars: Maximum characters per resume chunk.
        client: Optional LLMClient override passed through to
                llm.chat_with_resume().
        history: Rolling list of {"role": ..., "content": ...} chat
                 turns, used to give the LLM conversational context.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        top_k: int = DEFAULT_TOP_K,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
        client: Optional[LLMClient] = None,
    ) -> None:
        """
        Initialize the chatbot.

        Args:
            vector_store: An optional pre-constructed VectorStore
                           (e.g. one already loaded from disk via
                           VectorStore.load()). If not provided, a new
                           empty VectorStore is created.
            top_k: Number of chunks to retrieve per question.
            max_chunk_chars: Maximum characters per chunk when
                              ingesting resume text.
            client: Optional LLMClient instance to use for all LLM
                    calls. Uses llm.py's default client if not
                    provided.
        """
        self.vector_store: VectorStore = vector_store or VectorStore()
        self.top_k = top_k
        self.max_chunk_chars = max_chunk_chars
        self.client = client
        self.history: List[Dict[str, str]] = []

    # -------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------
    def ingest_resume(self, resume_text: str) -> int:
        """
        Chunk the given resume text and build/replace the vector
        index used for retrieval.

        Calling this again with new resume text replaces the
        previously indexed content and clears the chat history (since
        prior answers may no longer be relevant to the new resume).

        Args:
            resume_text: The full resume text to ingest (e.g. the
                          structured resume sections joined into one
                          string).

        Returns:
            The number of chunks that were indexed.

        Raises:
            ValueError: If `resume_text` is empty or whitespace-only,
                or if chunking produces no usable chunks.
            VectorStoreError: If embedding or index construction fails
                (propagated from VectorStore.build_index()).
        """
        if not resume_text or not resume_text.strip():
            raise ValueError("Cannot ingest empty resume text.")

        chunks = chunk_resume_text(resume_text, self.max_chunk_chars)
        if not chunks:
            raise ValueError("Resume text could not be split into any usable chunks.")

        logger.info("Ingesting resume: %d chunks to index.", len(chunks))

        # build_index() may raise VectorStoreError; let it propagate so
        # the caller (e.g. a Streamlit page) can show a clear error
        # rather than silently having an empty/stale index.
        self.vector_store.build_index(chunks)

        # A new resume invalidates any prior conversation context.
        self.history = []

        return len(chunks)

    # -------------------------------------------------------------------
    # Q&A
    # -------------------------------------------------------------------
    def ask(self, question: str) -> str:
        """
        Answer a question about the ingested resume using RAG.

        Workflow:
            1. Validate that a resume has been ingested and the
               question is non-empty.
            2. Retrieve the top-k most relevant resume chunks via
               VectorStore.search().
            3. Pass the question, retrieved chunks, and recent chat
               history to llm.chat_with_resume().
            4. Update the rolling chat history with this turn.

        Args:
            question: The user's question about their resume.

        Returns:
            The chatbot's answer as a string. If no resume has been
            ingested, or the question is empty, returns a
            user-friendly message instead of calling the LLM. If the
            LLM call itself fails, returns the
            "[LLM Error] ..." string produced by llm.chat_with_resume().
        """
        if not question or not question.strip():
            return "Please enter a question about your resume."

        if self.vector_store.is_empty():
            logger.warning("ask() called before a resume was ingested.")
            return (
                "No resume has been loaded yet. Please upload and process "
                "a resume before asking questions."
            )

        try:
            results = self.vector_store.search(question, top_k=self.top_k)
        except VectorStoreError as exc:
            logger.error("Vector store search failed: %s", exc)
            return f"[LLM Error] Failed to retrieve resume context: {exc}"

        context_chunks = [r["text"] for r in results if r.get("text")]

        if not context_chunks:
            logger.info("No relevant resume chunks found for question: %r", question)

        # Pass a bounded window of prior turns for conversational context.
        recent_history = self.history[-(MAX_HISTORY_TURNS * 2):]

        answer = chat_with_resume(
            question=question,
            context_chunks=context_chunks,
            chat_history=recent_history,
            client=self.client,
        )

        # Update rolling history (only after a successful-looking call;
        # we still record errors so the conversation log is complete,
        # but callers can inspect the "[LLM Error]" prefix themselves).
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        # Trim history to the configured window.
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

        return answer

    # -------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------
    def reset_conversation(self) -> None:
        """
        Clear the rolling chat history without affecting the indexed
        resume content.
        """
        self.history = []
        logger.debug("Chat history reset.")

    def is_ready(self) -> bool:
        """
        Check whether a resume has been ingested and the chatbot is
        ready to answer questions.

        Returns:
            True if the vector store contains indexed resume chunks,
            False otherwise.
        """
        return not self.vector_store.is_empty()


# ---------------------------------------------------------------------------
# Manual test entry point (not used by the Streamlit app directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    sample_resume_text = (
        "Jane Doe\n"
        "jane@example.com | +1 555 123 4567\n\n"
        "Skills\n"
        "Python, Machine Learning, FAISS, Streamlit, SQL\n\n"
        "Experience\n"
        "Data Science Intern, Acme Corp\n"
        "Worked on ML pipelines and NLP models, improving model "
        "accuracy by 12% through feature engineering.\n\n"
        "Projects\n"
        "Built a resume analyzer using NLP and FAISS for semantic "
        "search over resume content.\n\n"
        "Education\n"
        "B.Tech in Computer Science, 2024"
    )

    bot = ResumeChatbot()
    num_chunks = bot.ingest_resume(sample_resume_text)
    print(f"Indexed {num_chunks} chunks.")

    print(bot.ask("What programming languages does this candidate know?"))
    print(bot.ask("Tell me more about their most recent project."))

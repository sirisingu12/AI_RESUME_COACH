"""
embeddings.py
-------------
Module responsible for generating text embeddings using a local
sentence-transformers model (all-MiniLM-L6-v2).

Provides:
    - create_embedding(): embed a single piece of text.
    - create_embeddings_batch(): embed a list of texts efficiently in
      one batched call.

Design notes:
    - The model is loaded once per process using a singleton pattern,
      so repeated calls (e.g. from Streamlit re-runs) do not reload
      the model from disk each time.
    - All embedding computation happens locally; no network calls are
      made at inference time (aside from the one-time model
      download/cache by sentence-transformers on first use).
    - All functions handle empty input and model-loading/encoding
      errors gracefully, raising a clearly-typed exception
      (EmbeddingError) rather than letting raw library exceptions
      propagate unpredictably.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Name of the local sentence-transformers model used for all embeddings.
# Sourced from config.py so embeddings.py and jd_matcher.py always
# agree on which model is loaded.
MODEL_NAME: str = EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class EmbeddingError(Exception):
    """
    Raised when the embedding model fails to load or when text
    encoding fails for an unexpected reason.
    """


# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

# Module-level cache for the loaded model. Kept as a private variable
# rather than using functools.lru_cache so that load failures can be
# retried on a subsequent call (lru_cache would cache an exception-free
# call only after it succeeds, but we want explicit control here).
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """
    Get the shared sentence-transformers model instance, loading it
    from disk/cache on first use (singleton pattern).

    Returns:
        The loaded SentenceTransformer instance for MODEL_NAME.

    Raises:
        EmbeddingError: If the model fails to load (e.g. missing
            dependencies, corrupted cache, or no internet connection
            on first download).
    """
    global _model

    if _model is not None:
        return _model

    try:
        _model = SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to load embedding model '{MODEL_NAME}': {exc}"
        ) from exc

    return _model


# ---------------------------------------------------------------------------
# Public embedding functions
# ---------------------------------------------------------------------------
def create_embedding(text: str) -> List[float]:
    """
    Create an embedding vector for a single piece of text.

    Args:
        text: The text to embed. Leading/trailing whitespace is
              stripped before encoding.

    Returns:
        A list of floats representing the embedding vector.
        Returns an empty list if `text` is empty or whitespace-only.

    Raises:
        EmbeddingError: If the model fails to load or encoding fails.
    """
    if not text or not text.strip():
        return []

    model = get_model()

    try:
        vector: np.ndarray = model.encode(text.strip(), convert_to_numpy=True)
    except Exception as exc:
        raise EmbeddingError(f"Failed to create embedding for text: {exc}") from exc

    return vector.tolist()


def create_embeddings_batch(
    texts: List[str],
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> List[List[float]]:
    """
    Create embedding vectors for a batch of texts in one efficient
    call.

    Empty or whitespace-only strings in `texts` are preserved as empty
    vectors ([]) in the output, at the same positional index, so the
    output list always has the same length as the input list and
    callers can safely zip() the two together.

    Args:
        texts: A list of texts to embed.
        batch_size: Number of texts to encode per internal batch
                     (passed through to sentence-transformers).
        show_progress_bar: Whether to display a progress bar during
                            encoding (useful for large batches in a
                            script; usually False in a Streamlit app).

    Returns:
        A list of embedding vectors (each a list of floats), one per
        input text, in the same order. Texts that were empty or
        whitespace-only map to an empty list ([]).

        Returns an empty list if `texts` is empty.

    Raises:
        EmbeddingError: If the model fails to load or encoding fails.
    """
    if not texts:
        return []

    # Track which indices correspond to non-empty texts, since the
    # model should only be asked to encode meaningful text.
    non_empty_indices: List[int] = []
    non_empty_texts: List[str] = []

    for index, text in enumerate(texts):
        if text and text.strip():
            non_empty_indices.append(index)
            non_empty_texts.append(text.strip())

    # Pre-fill the result with empty vectors for every input position.
    results: List[List[float]] = [[] for _ in texts]

    if not non_empty_texts:
        return results

    model = get_model()

    try:
        vectors: np.ndarray = model.encode(
            non_empty_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
    except Exception as exc:
        raise EmbeddingError(f"Failed to create batch embeddings: {exc}") from exc

    for position, original_index in enumerate(non_empty_indices):
        results[original_index] = vectors[position].tolist()

    return results


# ---------------------------------------------------------------------------
# Manual test entry point (not used by the Streamlit app directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    single_vector = create_embedding("Python developer with FAISS experience")
    print(f"Single embedding length: {len(single_vector)}")

    batch_vectors = create_embeddings_batch(
        ["Python", "Machine Learning", "", "FAISS and Streamlit"]
    )
    for i, vec in enumerate(batch_vectors):
        print(f"Text {i}: embedding length = {len(vec)}")

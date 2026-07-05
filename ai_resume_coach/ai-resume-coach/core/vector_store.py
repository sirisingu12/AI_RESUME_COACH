"""
core/vector_store.py
---------------------
Module responsible for managing a local FAISS vector index over resume
text chunks, for use in retrieval-augmented generation (RAG).

Provides:
    - VectorStore: a class wrapping a FAISS index plus parallel chunk
      metadata, with build/search/save/load operations.

Compatible with:
    - embeddings.py (uses create_embedding / create_embeddings_batch
      for all embedding computation).
    - rag_chatbot.py (consumes VectorStore.search() results as
      retrieval context for the LLM).

Design notes:
    - Embeddings are L2-normalized and FAISS's inner-product index
      (IndexFlatIP) is used, so the inner product is equivalent to
      cosine similarity. Higher scores = more similar.
    - The index and its metadata are persisted as two files on disk:
      "<path>.index" (FAISS binary index) and "<path>.meta.json"
      (chunk text + metadata as JSON).
    - All operations are wrapped with error handling and logging.
      Missing index files on load() are handled gracefully (the store
      remains empty rather than raising).
    - No network calls are made; everything runs locally.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from embeddings import EmbeddingError, create_embedding, create_embeddings_batch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ChunkMetadata = Dict[str, Any]
SearchResult = Dict[str, Any]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class VectorStoreError(Exception):
    """
    Raised when index building, searching, saving, or loading fails
    for a reason that the caller should be aware of (as opposed to
    being silently handled).
    """


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------
class VectorStore:
    """
    A FAISS-backed vector store for resume text chunks.

    Stores:
        - A FAISS IndexFlatIP index of L2-normalized embeddings
          (inner product == cosine similarity for normalized vectors).
        - A parallel list of metadata dictionaries, one per indexed
          chunk, each containing at least the original chunk text
          under the "text" key.

    Attributes:
        index: The underlying FAISS index, or None if not yet built.
        metadata: List of metadata dicts, one per vector in `index`,
                  in the same order.
        dimension: The embedding dimension of the loaded model, set
                   automatically on the first call to build_index().
    """

    # File suffixes used for persistence.
    INDEX_SUFFIX = ".index"
    METADATA_SUFFIX = ".meta.json"

    def __init__(self) -> None:
        """
        Initialize an empty VectorStore.

        The FAISS index is not created until build_index() or load()
        is called, since the embedding dimension is not known ahead
        of time.
        """
        self.index: Optional[faiss.Index] = None
        self.metadata: List[ChunkMetadata] = []
        self.dimension: Optional[int] = None

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def is_empty(self) -> bool:
        """
        Check whether the store currently has no indexed vectors.

        Returns:
            True if the index has not been built/loaded, or contains
            zero vectors. False otherwise.
        """
        return self.index is None or self.index.ntotal == 0

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalize a 2D array of vectors in place-safe fashion.

        Args:
            vectors: A 2D numpy array of shape (n, dim), dtype float32.

        Returns:
            The same array, L2-normalized along axis 1. Rows that are
            all-zero are left unchanged (normalizing a zero vector
            would divide by zero).
        """
        # faiss.normalize_L2 operates in-place and skips zero vectors
        # safely (leaves them as zero), so it's used directly here.
        faiss.normalize_L2(vectors)
        return vectors

    # -------------------------------------------------------------------
    # 1. BUILD INDEX
    # -------------------------------------------------------------------
    def build_index(
        self,
        text_chunks: List[str],
        metadatas: Optional[List[ChunkMetadata]] = None,
    ) -> None:
        """
        Build a fresh FAISS index from a list of text chunks.

        Any existing index/metadata in this VectorStore is replaced.
        Empty or whitespace-only chunks are skipped (they would embed
        to empty vectors and provide no retrieval value).

        Args:
            text_chunks: A list of text chunks to embed and index
                          (e.g. resume sections or sentence-level
                          chunks).
            metadatas: Optional list of metadata dicts, one per entry
                       in `text_chunks`, in the same order. If
                       provided, must be the same length as
                       `text_chunks`. Each dict is merged with
                       {"text": chunk, "chunk_id": index}. If not
                       provided, metadata defaults to just
                       {"text": chunk, "chunk_id": index}.

        Raises:
            VectorStoreError: If `text_chunks` is empty, if
                `metadatas` has a mismatched length, if embedding
                fails, or if the resulting embeddings are empty.
        """
        if not text_chunks:
            raise VectorStoreError("Cannot build an index from an empty list of text chunks.")

        if metadatas is not None and len(metadatas) != len(text_chunks):
            raise VectorStoreError(
                "Length of 'metadatas' must match length of 'text_chunks' "
                f"(got {len(metadatas)} and {len(text_chunks)})."
            )

        logger.info("Building vector index from %d text chunks.", len(text_chunks))

        # Filter out empty/whitespace-only chunks, keeping metadata aligned.
        filtered_chunks: List[str] = []
        filtered_metadatas: List[ChunkMetadata] = []

        for i, chunk in enumerate(text_chunks):
            if chunk and chunk.strip():
                filtered_chunks.append(chunk.strip())
                base_meta: ChunkMetadata = dict(metadatas[i]) if metadatas else {}
                base_meta.setdefault("chunk_id", i)
                base_meta["text"] = chunk.strip()
                filtered_metadatas.append(base_meta)
            else:
                logger.debug("Skipping empty chunk at position %d.", i)

        if not filtered_chunks:
            raise VectorStoreError(
                "All provided text chunks were empty after stripping whitespace."
            )

        try:
            embeddings = create_embeddings_batch(filtered_chunks)
        except EmbeddingError as exc:
            raise VectorStoreError(f"Failed to embed text chunks: {exc}") from exc

        vectors = np.array(embeddings, dtype="float32")

        if vectors.ndim != 2 or vectors.shape[0] != len(filtered_chunks):
            raise VectorStoreError(
                "Embedding output shape is invalid; expected "
                f"({len(filtered_chunks)}, dim) but got {vectors.shape}."
            )

        dimension = vectors.shape[1]
        self._normalize(vectors)

        try:
            index = faiss.IndexFlatIP(dimension)
            index.add(vectors)
        except Exception as exc:
            raise VectorStoreError(f"Failed to build FAISS index: {exc}") from exc

        self.index = index
        self.metadata = filtered_metadatas
        self.dimension = dimension

        logger.info(
            "Vector index built successfully: %d vectors, dimension=%d.",
            self.index.ntotal,
            self.dimension,
        )

    # -------------------------------------------------------------------
    # 2. SEARCH
    # -------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search the index for the chunks most similar to `query`.

        Args:
            query: The query text (e.g. a user's chatbot question).
            top_k: Maximum number of results to return.

        Returns:
            A list of result dictionaries, ordered by descending
            similarity, each with:
                - "text": str — the chunk's text.
                - "score": float — cosine similarity score
                  (higher = more similar; range approximately
                  [-1, 1], typically [0, 1] for natural text).
                - "metadata": Dict[str, Any] — the full metadata dict
                  for this chunk (including "text" and "chunk_id").

            Returns an empty list if:
                - The store is empty (no index built/loaded).
                - `query` is empty or whitespace-only.
                - `top_k` is less than 1.

        Raises:
            VectorStoreError: If embedding the query fails or the
                FAISS search itself raises an unexpected error.
        """
        if self.is_empty():
            logger.warning("search() called on an empty VectorStore; returning no results.")
            return []

        if not query or not query.strip():
            logger.debug("search() called with empty query; returning no results.")
            return []

        if top_k < 1:
            logger.debug("search() called with top_k < 1; returning no results.")
            return []

        try:
            query_vector_list = create_embedding(query)
        except EmbeddingError as exc:
            raise VectorStoreError(f"Failed to embed search query: {exc}") from exc

        if not query_vector_list:
            logger.debug("Query embedding was empty; returning no results.")
            return []

        query_vector = np.array([query_vector_list], dtype="float32")
        self._normalize(query_vector)

        # Don't request more neighbors than vectors available.
        effective_k = min(top_k, self.index.ntotal)

        try:
            scores, indices = self.index.search(query_vector, effective_k)
        except Exception as exc:
            raise VectorStoreError(f"FAISS search failed: {exc}") from exc

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                # FAISS returns -1 for unfilled slots when there are
                # fewer than `effective_k` results available.
                continue

            meta = self.metadata[idx]
            results.append(
                {
                    "text": meta.get("text", ""),
                    "score": float(score),
                    "metadata": meta,
                }
            )

        logger.debug("search() returned %d results for query of length %d.", len(results), len(query))
        return results

    # -------------------------------------------------------------------
    # 3. SAVE
    # -------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Persist the FAISS index and chunk metadata to disk.

        Two files are written:
            - "<path>.index": the binary FAISS index.
            - "<path>.meta.json": a JSON file containing the chunk
              metadata list and the embedding dimension.

        Any directories in `path` are created if they do not exist.

        Args:
            path: Base path (without extension) to save to, e.g.
                  "data/faiss_index/resume_index".

        Raises:
            VectorStoreError: If the store is empty (nothing to save),
                or if writing either file fails.
        """
        if self.is_empty():
            raise VectorStoreError("Cannot save an empty VectorStore (no index has been built).")

        directory = os.path.dirname(path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                raise VectorStoreError(f"Failed to create directory '{directory}': {exc}") from exc

        index_path = path + self.INDEX_SUFFIX
        metadata_path = path + self.METADATA_SUFFIX

        try:
            faiss.write_index(self.index, index_path)
        except Exception as exc:
            raise VectorStoreError(f"Failed to write FAISS index to '{index_path}': {exc}") from exc

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"dimension": self.dimension, "metadata": self.metadata},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except (OSError, TypeError) as exc:
            raise VectorStoreError(f"Failed to write metadata to '{metadata_path}': {exc}") from exc

        logger.info(
            "Vector store saved: %d vectors -> '%s' and '%s'.",
            self.index.ntotal,
            index_path,
            metadata_path,
        )

    # -------------------------------------------------------------------
    # 4. LOAD
    # -------------------------------------------------------------------
    def load(self, path: str) -> bool:
        """
        Load a previously saved FAISS index and metadata from disk.

        If either expected file is missing, the store is left empty
        (or unchanged, if already populated) and a warning is logged
        rather than an exception being raised — this allows
        rag_chatbot.py to handle "no index yet" as a normal first-run
        condition.

        Args:
            path: Base path (without extension) to load from, e.g.
                  "data/faiss_index/resume_index". Must match the path
                  previously passed to save().

        Returns:
            True if the index and metadata were loaded successfully,
            False if either file was missing (store left empty).

        Raises:
            VectorStoreError: If the files exist but are corrupted or
                cannot be parsed.
        """
        index_path = path + self.INDEX_SUFFIX
        metadata_path = path + self.METADATA_SUFFIX

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning(
                "No existing vector index found at '%s' (looked for '%s' and '%s'). "
                "Store remains empty.",
                path,
                index_path,
                metadata_path,
            )
            return False

        try:
            index = faiss.read_index(index_path)
        except Exception as exc:
            raise VectorStoreError(f"Failed to read FAISS index from '{index_path}': {exc}") from exc

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Failed to read metadata from '{metadata_path}': {exc}") from exc

        metadata = payload.get("metadata")
        dimension = payload.get("dimension")

        if not isinstance(metadata, list):
            raise VectorStoreError(
                f"Metadata file '{metadata_path}' is malformed: 'metadata' is not a list."
            )

        if index.ntotal != len(metadata):
            raise VectorStoreError(
                f"Loaded index has {index.ntotal} vectors but metadata has "
                f"{len(metadata)} entries; files may be out of sync."
            )

        self.index = index
        self.metadata = metadata
        self.dimension = dimension

        logger.info(
            "Vector store loaded: %d vectors from '%s' and '%s'.",
            self.index.ntotal,
            index_path,
            metadata_path,
        )
        return True


# ---------------------------------------------------------------------------
# Manual test entry point (not used by the Streamlit app directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    chunks = [
        "Skills: Python, Machine Learning, FAISS, Streamlit, SQL",
        "Experience: Data Science Intern at Acme Corp, worked on ML pipelines and NLP models.",
        "Projects: Built a resume analyzer using NLP and vector search.",
        "Education: B.Tech in Computer Science, 2024",
    ]

    store = VectorStore()
    store.build_index(chunks)

    results = store.search("What programming languages does this candidate know?", top_k=2)
    for r in results:
        print(f"score={r['score']:.4f} -> {r['text']}")

    store.save("data/faiss_index/demo_index")

    new_store = VectorStore()
    loaded = new_store.load("data/faiss_index/demo_index")
    print(f"Loaded: {loaded}, vectors: {new_store.index.ntotal if new_store.index else 0}")

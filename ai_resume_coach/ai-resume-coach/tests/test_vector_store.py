"""
tests/test_vector_store.py
----------------------------
Unit tests for core/vector_store.py.

The embedding model is mocked in all tests so no GPU or downloaded
model is needed. FAISS itself is real — only the embedding call is
patched.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.vector_store import VectorStore, VectorStoreError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_embeddings_batch(texts):
    """
    Return deterministic 384-dim unit vectors for each text.
    Different texts get different vectors via a hash-based offset.
    """
    vecs = []
    for i, _ in enumerate(texts):
        vec = np.zeros(384, dtype="float32")
        vec[i % 384] = 1.0
        vecs.append(vec.tolist())
    return vecs


def _fake_embedding(text):
    vec = np.ones(384, dtype="float32")
    vec /= np.linalg.norm(vec)
    return vec.tolist()


# ---------------------------------------------------------------------------
# VectorStore.build_index
# ---------------------------------------------------------------------------
class TestBuildIndex(unittest.TestCase):

    @patch("core.vector_store.create_embeddings_batch", side_effect=_fake_embeddings_batch)
    def test_build_index_populates_store(self, _):
        store = VectorStore()
        store.build_index(["Python developer", "Machine learning engineer"])
        self.assertFalse(store.is_empty())
        self.assertEqual(store.index.ntotal, 2)

    @patch("core.vector_store.create_embeddings_batch", side_effect=_fake_embeddings_batch)
    def test_metadata_length_matches_chunks(self, _):
        store = VectorStore()
        chunks = ["Skills: Python", "Education: BSc", "Experience: intern"]
        store.build_index(chunks)
        self.assertEqual(len(store.metadata), len(chunks))

    @patch("core.vector_store.create_embeddings_batch", side_effect=_fake_embeddings_batch)
    def test_empty_chunks_are_skipped(self, _):
        store = VectorStore()
        store.build_index(["Good chunk", "", "  ", "Another good chunk"])
        self.assertEqual(store.index.ntotal, 2)

    def test_empty_list_raises(self):
        store = VectorStore()
        with self.assertRaises(VectorStoreError):
            store.build_index([])

    def test_all_empty_strings_raises(self):
        store = VectorStore()
        with self.assertRaises(VectorStoreError):
            store.build_index(["", "   "])

    @patch("core.vector_store.create_embeddings_batch", side_effect=_fake_embeddings_batch)
    def test_mismatched_metadata_length_raises(self, _):
        store = VectorStore()
        with self.assertRaises(VectorStoreError):
            store.build_index(["chunk1", "chunk2"], metadatas=[{"a": 1}])


# ---------------------------------------------------------------------------
# VectorStore.search
# ---------------------------------------------------------------------------
class TestSearch(unittest.TestCase):

    def _build_store(self):
        with patch("core.vector_store.create_embeddings_batch", side_effect=_fake_embeddings_batch):
            store = VectorStore()
            store.build_index([
                "Skills: Python, Machine Learning",
                "Education: B.Tech Computer Science",
                "Projects: Resume Analyzer using NLP",
            ])
        return store

    @patch("core.vector_store.create_embedding", side_effect=_fake_embedding)
    def test_search_returns_list(self, _):
        store = self._build_store()
        results = store.search("Python skills", top_k=2)
        self.assertIsInstance(results, list)

    @patch("core.vector_store.create_embedding", side_effect=_fake_embedding)
    def test_search_respects_top_k(self, _):
        store = self._build_store()
        results = store.search("Python skills", top_k=2)
        self.assertLessEqual(len(results), 2)

    @patch("core.vector_store.create_embedding", side_effect=_fake_embedding)
    def test_result_schema(self, _):
        store = self._build_store()
        results = store.search("skills", top_k=1)
        if results:
            self.assertIn("text", results[0])
            self.assertIn("score", results[0])
            self.assertIn("metadata", results[0])

    def test_search_on_empty_store_returns_empty(self):
        store = VectorStore()
        results = store.search("anything", top_k=3)
        self.assertEqual(results, [])

    @patch("core.vector_store.create_embedding", side_effect=_fake_embedding)
    def test_empty_query_returns_empty(self, _):
        store = self._build_store()
        results = store.search("", top_k=3)
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# VectorStore.save and .load
# ---------------------------------------------------------------------------
class TestSaveLoad(unittest.TestCase):

    @patch("core.vector_store.create_embeddings_batch", side_effect=_fake_embeddings_batch)
    def test_save_creates_files(self, _):
        store = VectorStore()
        store.build_index(["Python developer", "Machine learning"])

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "test_index")
            store.save(base_path)
            self.assertTrue(os.path.exists(base_path + ".index"))
            self.assertTrue(os.path.exists(base_path + ".meta.json"))

    @patch("core.vector_store.create_embeddings_batch", side_effect=_fake_embeddings_batch)
    def test_save_then_load_roundtrip(self, _):
        store = VectorStore()
        chunks = ["Python skills", "ML experience", "NLP projects"]
        store.build_index(chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "test_index")
            store.save(base_path)

            new_store = VectorStore()
            loaded = new_store.load(base_path)

            self.assertTrue(loaded)
            self.assertEqual(new_store.index.ntotal, store.index.ntotal)
            self.assertEqual(len(new_store.metadata), len(store.metadata))

    def test_load_missing_files_returns_false(self):
        store = VectorStore()
        result = store.load("/tmp/definitely_does_not_exist_xyz/index")
        self.assertFalse(result)
        self.assertTrue(store.is_empty())

    def test_save_empty_store_raises(self):
        store = VectorStore()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(VectorStoreError):
                store.save(os.path.join(tmpdir, "empty"))


# ---------------------------------------------------------------------------
# VectorStore.is_empty
# ---------------------------------------------------------------------------
class TestIsEmpty(unittest.TestCase):

    def test_new_store_is_empty(self):
        self.assertTrue(VectorStore().is_empty())

    @patch("core.vector_store.create_embeddings_batch", side_effect=_fake_embeddings_batch)
    def test_store_not_empty_after_build(self, _):
        store = VectorStore()
        store.build_index(["some text"])
        self.assertFalse(store.is_empty())


if __name__ == "__main__":
    unittest.main()

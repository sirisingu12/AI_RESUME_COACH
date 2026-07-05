"""
tests/test_jd_matcher.py
--------------------------
Unit tests for core/jd_matcher.py.

Embedding model tests mock the SentenceTransformer to avoid requiring
a GPU / downloaded model during CI. A small number of tests use a
real model and are marked with @unittest.skip by default — remove the
skip decorator to run them locally with the model installed.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.jd_matcher import (
    extract_keywords,
    calculate_semantic_similarity,
    detect_matched_and_missing_skills,
    match_resume_to_job,
)


RESUME_TEXT = (
    "Skills: Python, Machine Learning, FAISS, Streamlit, SQL\n"
    "Experience: Data Science Intern, worked on NLP pipelines.\n"
    "Projects: Built a resume analyzer using NLP."
)

JD_TEXT = (
    "Looking for a Python developer with experience in machine learning, "
    "NLP, FAISS, and Streamlit. Docker knowledge is a plus."
)


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------
class TestExtractKeywords(unittest.TestCase):

    def test_returns_list(self):
        result = extract_keywords(JD_TEXT)
        self.assertIsInstance(result, list)

    def test_no_duplicates(self):
        result = extract_keywords("python python java")
        self.assertEqual(len(result), len(set(result)))

    def test_lowercased(self):
        result = extract_keywords("Python Java")
        self.assertIn("python", result)

    def test_empty_text_returns_empty(self):
        self.assertEqual(extract_keywords(""), [])

    def test_stopwords_excluded(self):
        result = extract_keywords("we are looking for a strong candidate")
        self.assertNotIn("we", result)
        self.assertNotIn("for", result)


# ---------------------------------------------------------------------------
# calculate_semantic_similarity (mocked model)
# ---------------------------------------------------------------------------
class TestCalculateSemanticSimilarity(unittest.TestCase):

    def test_empty_resume_returns_zero(self):
        with patch("core.jd_matcher.get_model") as mock_get:
            score = calculate_semantic_similarity("", JD_TEXT)
        self.assertEqual(score, 0.0)

    def test_empty_jd_returns_zero(self):
        with patch("core.jd_matcher.get_model"):
            score = calculate_semantic_similarity(RESUME_TEXT, "")
        self.assertEqual(score, 0.0)

    def test_score_in_range(self):
        # Use a mock that returns unit vectors (cosine sim = 1.0).
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            __getitem__=lambda self, i: np.array([1.0, 0.0, 0.0])
        )
        with patch("core.jd_matcher.get_model", return_value=mock_model):
            score = calculate_semantic_similarity(RESUME_TEXT, JD_TEXT)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)


# ---------------------------------------------------------------------------
# match_resume_to_job (integration — output schema)
# ---------------------------------------------------------------------------
class TestMatchResumeToJob(unittest.TestCase):

    def _mock_model(self):
        """Return a mock SentenceTransformer that produces random unit vectors."""
        mock = MagicMock()
        def fake_encode(text_or_list, **kwargs):
            if isinstance(text_or_list, list):
                n = len(text_or_list)
                vecs = np.random.randn(n, 384).astype("float32")
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / np.where(norms == 0, 1, norms)
            else:
                vec = np.random.randn(384).astype("float32")
                return vec / (np.linalg.norm(vec) or 1)
        mock.encode.side_effect = fake_encode
        return mock

    def test_output_schema_complete(self):
        with patch("core.jd_matcher.get_model", return_value=self._mock_model()):
            result = match_resume_to_job(RESUME_TEXT, JD_TEXT)
        for key in ("match_score", "matched_skills", "missing_skills"):
            self.assertIn(key, result)

    def test_match_score_in_range(self):
        with patch("core.jd_matcher.get_model", return_value=self._mock_model()):
            result = match_resume_to_job(RESUME_TEXT, JD_TEXT)
        self.assertGreaterEqual(result["match_score"], 0.0)
        self.assertLessEqual(result["match_score"], 100.0)

    def test_matched_and_missing_are_lists(self):
        with patch("core.jd_matcher.get_model", return_value=self._mock_model()):
            result = match_resume_to_job(RESUME_TEXT, JD_TEXT)
        self.assertIsInstance(result["matched_skills"], list)
        self.assertIsInstance(result["missing_skills"], list)

    def test_empty_inputs_return_zero(self):
        with patch("core.jd_matcher.get_model", return_value=self._mock_model()):
            result = match_resume_to_job("", "")
        self.assertEqual(result["match_score"], 0.0)


if __name__ == "__main__":
    unittest.main()

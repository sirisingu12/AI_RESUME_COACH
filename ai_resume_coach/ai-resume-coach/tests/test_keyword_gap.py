"""
tests/test_keyword_gap.py
--------------------------
Unit tests for core/keyword_gap.py.

No external dependencies (no LLM, no embeddings) — fully offline.
"""

from __future__ import annotations

import unittest

from core.keyword_gap import (
    analyze_keyword_gap,
    extract_keywords,
)


RESUME_TEXT = (
    "Skills: Python, Machine Learning, FAISS, Streamlit, SQL\n"
    "Experience: Data Science Intern at Acme Corp, worked on NLP models.\n"
    "Projects: Resume analyzer using vector search."
)

JD_TEXT = (
    "We need a Python developer with machine learning, NLP, FAISS, "
    "Streamlit. Docker knowledge is a plus."
)


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------
class TestExtractKeywords(unittest.TestCase):

    def test_returns_list(self):
        self.assertIsInstance(extract_keywords(JD_TEXT), list)

    def test_no_duplicates(self):
        kws = extract_keywords("python python java java")
        self.assertEqual(len(kws), len(set(kws)))

    def test_stopwords_removed(self):
        kws = extract_keywords("we are looking for a developer")
        self.assertNotIn("we", kws)
        self.assertNotIn("are", kws)

    def test_empty_string_returns_empty(self):
        self.assertEqual(extract_keywords(""), [])

    def test_preserves_tech_symbols(self):
        kws = extract_keywords("c++ and c# development")
        kw_string = " ".join(kws)
        self.assertTrue("c++" in kw_string or "c#" in kw_string)

    def test_pipe_delimiter_splits_skills(self):
        kws = extract_keywords("Python | FAISS | Streamlit")
        self.assertIn("python", kws)
        self.assertIn("faiss", kws)


# ---------------------------------------------------------------------------
# analyze_keyword_gap
# ---------------------------------------------------------------------------
class TestAnalyzeKeywordGap(unittest.TestCase):

    def test_output_schema_complete(self):
        result = analyze_keyword_gap(RESUME_TEXT, JD_TEXT)
        for key in ("present_keywords", "missing_keywords",
                    "total_keywords", "match_percentage"):
            self.assertIn(key, result)

    def test_present_and_missing_partition_total(self):
        result = analyze_keyword_gap(RESUME_TEXT, JD_TEXT)
        self.assertEqual(
            len(result["present_keywords"]) + len(result["missing_keywords"]),
            result["total_keywords"],
        )

    def test_match_percentage_in_range(self):
        result = analyze_keyword_gap(RESUME_TEXT, JD_TEXT)
        self.assertGreaterEqual(result["match_percentage"], 0.0)
        self.assertLessEqual(result["match_percentage"], 100.0)

    def test_python_present_in_result(self):
        result = analyze_keyword_gap(RESUME_TEXT, JD_TEXT)
        present_lower = [k.lower() for k in result["present_keywords"]]
        self.assertTrue(
            any("python" in k for k in present_lower),
            "Expected 'python' in present_keywords"
        )

    def test_docker_missing_from_resume(self):
        result = analyze_keyword_gap(RESUME_TEXT, JD_TEXT)
        missing_lower = [k.lower() for k in result["missing_keywords"]]
        self.assertTrue(
            any("docker" in k for k in missing_lower),
            "Expected 'docker' in missing_keywords"
        )

    def test_empty_jd_returns_zeroed_result(self):
        result = analyze_keyword_gap(RESUME_TEXT, "")
        self.assertEqual(result["total_keywords"], 0)
        self.assertEqual(result["match_percentage"], 0.0)
        self.assertEqual(result["present_keywords"], [])
        self.assertEqual(result["missing_keywords"], [])

    def test_empty_resume_all_keywords_missing(self):
        result = analyze_keyword_gap("", JD_TEXT)
        self.assertEqual(len(result["present_keywords"]), 0)
        self.assertEqual(result["match_percentage"], 0.0)

    def test_identical_texts_high_match(self):
        result = analyze_keyword_gap(JD_TEXT, JD_TEXT)
        self.assertGreater(result["match_percentage"], 70.0)

    def test_match_percentage_computation(self):
        result = analyze_keyword_gap(RESUME_TEXT, JD_TEXT)
        expected = round(
            len(result["present_keywords"]) / result["total_keywords"] * 100, 2
        )
        self.assertAlmostEqual(result["match_percentage"], expected, places=1)


if __name__ == "__main__":
    unittest.main()

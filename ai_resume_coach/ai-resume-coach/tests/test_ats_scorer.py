"""
tests/test_ats_scorer.py
-------------------------
Unit tests for core/ats_scorer.py.

No LLM or embedding model is needed — ats_scorer is purely
text/keyword based.
"""

from __future__ import annotations

import unittest

from core.ats_scorer import (
    calculate_ats_score,
    calculate_section_match_score,
    calculate_structure_score,
    extract_keywords,
    find_missing_keywords,
)
from config import ATS_WEIGHTS


FULL_RESUME = {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "phone": "+1 555 123 4567",
    "skills": ["Python", "Machine Learning", "FAISS", "Streamlit", "SQL"],
    "education": ["B.Tech in Computer Science, 2024"],
    "projects": ["Built a resume analyzer using NLP and FAISS"],
    "experience": ["Data Science Intern at Acme Corp, worked on ML pipelines"],
}

EMPTY_RESUME = {
    "name": "",
    "email": "",
    "phone": "",
    "skills": [],
    "education": [],
    "projects": [],
    "experience": [],
}


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------
class TestExtractKeywords(unittest.TestCase):

    def test_returns_set(self):
        result = extract_keywords("Python developer with FAISS")
        self.assertIsInstance(result, set)

    def test_lowercases_tokens(self):
        result = extract_keywords("Python FAISS")
        self.assertIn("python", result)

    def test_removes_stopwords(self):
        result = extract_keywords("experience with the team")
        self.assertNotIn("the", result)
        self.assertNotIn("with", result)

    def test_empty_text_returns_empty_set(self):
        self.assertEqual(extract_keywords(""), set())


# ---------------------------------------------------------------------------
# calculate_section_match_score
# ---------------------------------------------------------------------------
class TestSectionMatchScore(unittest.TestCase):

    def test_perfect_match_returns_max_points(self):
        jd_kws = {"python", "faiss", "sql"}
        score = calculate_section_match_score("python faiss sql", jd_kws, max_points=40)
        self.assertAlmostEqual(score, 40.0, delta=1.0)

    def test_zero_match_returns_zero(self):
        jd_kws = {"docker", "kubernetes"}
        score = calculate_section_match_score("python sql", jd_kws, max_points=40)
        self.assertEqual(score, 0.0)

    def test_empty_jd_keywords_returns_max(self):
        score = calculate_section_match_score("anything", set(), max_points=40)
        self.assertEqual(score, 40.0)

    def test_empty_section_returns_zero(self):
        score = calculate_section_match_score("", {"python"}, max_points=40)
        self.assertEqual(score, 0.0)

    def test_score_within_bounds(self):
        score = calculate_section_match_score("python ml", {"python", "docker"}, max_points=25)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 25.0)


# ---------------------------------------------------------------------------
# calculate_structure_score
# ---------------------------------------------------------------------------
class TestStructureScore(unittest.TestCase):

    def test_full_resume_gets_full_structure_score(self):
        score = calculate_structure_score(FULL_RESUME)
        self.assertEqual(score, ATS_WEIGHTS["structure"])

    def test_empty_resume_gets_zero_structure_score(self):
        score = calculate_structure_score(EMPTY_RESUME)
        self.assertEqual(score, 0.0)

    def test_partial_resume_gets_partial_score(self):
        partial = {**EMPTY_RESUME, "name": "Jane", "email": "jane@example.com"}
        score = calculate_structure_score(partial)
        self.assertGreater(score, 0.0)
        self.assertLess(score, ATS_WEIGHTS["structure"])


# ---------------------------------------------------------------------------
# find_missing_keywords
# ---------------------------------------------------------------------------
class TestFindMissingKeywords(unittest.TestCase):

    def test_keyword_present_in_resume(self):
        jd_kws = {"python", "docker"}
        missing = find_missing_keywords(
            {"skills": ["Python"], "education": [], "projects": [], "experience": []},
            jd_kws,
        )
        self.assertNotIn("python", missing)

    def test_keyword_absent_from_resume(self):
        jd_kws = {"docker"}
        missing = find_missing_keywords(
            {"skills": ["Python"], "education": [], "projects": [], "experience": []},
            jd_kws,
        )
        self.assertIn("docker", missing)

    def test_empty_jd_keywords_returns_empty_list(self):
        result = find_missing_keywords(FULL_RESUME, set())
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# calculate_ats_score (integration)
# ---------------------------------------------------------------------------
class TestCalculateATSScore(unittest.TestCase):

    JD = (
        "Looking for a Python developer with machine learning, FAISS, "
        "Streamlit, and SQL experience."
    )

    def test_output_schema_complete(self):
        result = calculate_ats_score(FULL_RESUME, self.JD)
        for key in ("overall_score", "skills_score", "experience_score",
                    "projects_score", "education_score", "structure_score",
                    "missing_keywords"):
            self.assertIn(key, result)

    def test_overall_score_in_range(self):
        result = calculate_ats_score(FULL_RESUME, self.JD)
        self.assertGreaterEqual(result["overall_score"], 0)
        self.assertLessEqual(result["overall_score"], 100)

    def test_empty_resume_gets_low_score(self):
        result = calculate_ats_score(EMPTY_RESUME, self.JD)
        self.assertLess(result["overall_score"], 20)

    def test_empty_jd_all_section_scores_are_max(self):
        result = calculate_ats_score(FULL_RESUME, "")
        self.assertEqual(result["skills_score"], ATS_WEIGHTS["skills"])
        self.assertEqual(result["experience_score"], ATS_WEIGHTS["experience"])

    def test_missing_keywords_is_list(self):
        result = calculate_ats_score(FULL_RESUME, self.JD)
        self.assertIsInstance(result["missing_keywords"], list)

    def test_weights_sum_matches_max_possible(self):
        result = calculate_ats_score(FULL_RESUME, "")
        total = sum(ATS_WEIGHTS.values())
        self.assertEqual(total, 100)


if __name__ == "__main__":
    unittest.main()

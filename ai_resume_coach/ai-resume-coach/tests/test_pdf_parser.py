"""
tests/test_pdf_parser.py
-------------------------
Unit tests for core/pdf_parser.py.

Tests are intentionally dependency-light: they mock PdfReader so no
real PDF files are required to run the test suite.
"""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

from core.pdf_parser import (
    clean_text,
    extract_email,
    extract_name,
    extract_phone,
    extract_text_from_pdf,
    parse_resume,
    split_into_sections,
    extract_skills,
    extract_education,
    extract_experience,
    extract_projects,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------
class TestCleanText(unittest.TestCase):

    def test_collapses_multiple_spaces(self):
        self.assertEqual(clean_text("hello   world"), "hello world")

    def test_strips_blank_lines(self):
        result = clean_text("line1\n\n\nline2")
        self.assertNotIn("\n\n", result)

    def test_empty_string_returns_empty(self):
        self.assertEqual(clean_text(""), "")

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(clean_text("   \n\t  "), "")

    def test_preserves_newlines_between_content(self):
        result = clean_text("Skills\nPython, Java")
        self.assertIn("Skills", result)
        self.assertIn("Python", result)


# ---------------------------------------------------------------------------
# extract_email
# ---------------------------------------------------------------------------
class TestExtractEmail(unittest.TestCase):

    def test_standard_email(self):
        self.assertEqual(extract_email("Contact: jane@example.com"), "jane@example.com")

    def test_no_email_returns_empty(self):
        self.assertEqual(extract_email("No email here"), "")

    def test_email_with_subdomain(self):
        self.assertIn("@", extract_email("jane@mail.example.com"))

    def test_empty_text(self):
        self.assertEqual(extract_email(""), "")


# ---------------------------------------------------------------------------
# extract_phone
# ---------------------------------------------------------------------------
class TestExtractPhone(unittest.TestCase):

    def test_ten_digit_phone(self):
        result = extract_phone("Call me at 123-456-7890")
        self.assertTrue(len([c for c in result if c.isdigit()]) >= 10)

    def test_phone_with_country_code(self):
        result = extract_phone("+1 555 123 4567")
        self.assertTrue(len([c for c in result if c.isdigit()]) >= 10)

    def test_no_phone_returns_empty(self):
        self.assertEqual(extract_phone("No phone number here"), "")

    def test_year_not_matched_as_phone(self):
        # "2024" has only 4 digits, should not be returned as a phone number.
        result = extract_phone("Graduated in 2024")
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# extract_name
# ---------------------------------------------------------------------------
class TestExtractName(unittest.TestCase):

    def test_extracts_name_from_first_line(self):
        text = "Jane Doe\njane@example.com\nSkills"
        self.assertEqual(extract_name(text), "Jane Doe")

    def test_skips_email_line(self):
        text = "jane@example.com\nJane Doe\nSkills"
        self.assertNotEqual(extract_name(text), "jane@example.com")

    def test_skips_section_header(self):
        text = "Skills\nPython\nJane Doe"
        self.assertNotEqual(extract_name(text), "Skills")

    def test_empty_text_returns_empty(self):
        self.assertEqual(extract_name(""), "")


# ---------------------------------------------------------------------------
# split_into_sections
# ---------------------------------------------------------------------------
class TestSplitIntoSections(unittest.TestCase):

    SAMPLE = (
        "Jane Doe\njane@example.com\n\n"
        "Skills\nPython, SQL\n\n"
        "Education\nB.Tech CSE 2024\n\n"
        "Experience\nIntern at Acme Corp\n\n"
        "Projects\nResume Analyzer\n"
    )

    def test_skills_section_detected(self):
        sections = split_into_sections(self.SAMPLE)
        self.assertIn("Python", " ".join(sections["skills"]))

    def test_education_section_detected(self):
        sections = split_into_sections(self.SAMPLE)
        self.assertIn("B.Tech", " ".join(sections["education"]))

    def test_missing_section_returns_empty_list(self):
        sections = split_into_sections("Name\nJane Doe")
        self.assertEqual(sections["certifications"], [])

    def test_all_keys_present_even_if_empty(self):
        sections = split_into_sections("")
        for key in ("skills", "education", "experience", "projects", "certifications"):
            self.assertIn(key, sections)


# ---------------------------------------------------------------------------
# extract_skills
# ---------------------------------------------------------------------------
class TestExtractSkills(unittest.TestCase):

    def test_comma_separated_skills(self):
        skills = extract_skills(["Python, Java, SQL"])
        self.assertIn("Python", skills)
        self.assertIn("Java", skills)

    def test_pipe_separated_skills(self):
        skills = extract_skills(["Python | FAISS | Streamlit"])
        self.assertEqual(len(skills), 3)

    def test_deduplication(self):
        skills = extract_skills(["Python, Python, Java"])
        self.assertEqual(skills.count("Python"), 1)

    def test_empty_input(self):
        self.assertEqual(extract_skills([]), [])


# ---------------------------------------------------------------------------
# parse_resume (integration)
# ---------------------------------------------------------------------------
class TestParseResume(unittest.TestCase):

    SAMPLE_TEXT = (
        "Jane Doe\n"
        "jane@example.com\n"
        "+1 555 123 4567\n\n"
        "Skills\n"
        "Python, Machine Learning, SQL\n\n"
        "Education\n"
        "B.Tech Computer Science 2024\n\n"
        "Experience\n"
        "Data Science Intern at Acme Corp\n\n"
        "Projects\n"
        "Resume Analyzer using NLP and FAISS\n"
    )

    def _mock_reader(self, text: str):
        """Build a mock PdfReader whose single page returns `text`."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = text
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.is_encrypted = False
        return mock_reader

    @patch("core.pdf_parser.PdfReader")
    def test_email_extracted(self, mock_pdf_reader):
        mock_pdf_reader.return_value = self._mock_reader(self.SAMPLE_TEXT)
        result = parse_resume(io.BytesIO(b"fake"))
        self.assertEqual(result["email"], "jane@example.com")

    @patch("core.pdf_parser.PdfReader")
    def test_skills_extracted(self, mock_pdf_reader):
        mock_pdf_reader.return_value = self._mock_reader(self.SAMPLE_TEXT)
        result = parse_resume(io.BytesIO(b"fake"))
        self.assertTrue(len(result["skills"]) > 0)

    @patch("core.pdf_parser.PdfReader")
    def test_empty_pdf_returns_default_structure(self, mock_pdf_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.is_encrypted = False
        mock_pdf_reader.return_value = mock_reader
        result = parse_resume(io.BytesIO(b"fake"))
        self.assertEqual(result["skills"], [])
        self.assertEqual(result["email"], "")

    @patch("core.pdf_parser.PdfReader")
    def test_malformed_pdf_returns_default_structure(self, mock_pdf_reader):
        from pypdf.errors import PdfReadError
        mock_pdf_reader.side_effect = PdfReadError("bad file")
        result = parse_resume(io.BytesIO(b"not a pdf"))
        self.assertEqual(result["name"], "")
        self.assertEqual(result["skills"], [])

    @patch("core.pdf_parser.PdfReader")
    def test_output_schema_always_complete(self, mock_pdf_reader):
        mock_pdf_reader.side_effect = Exception("unexpected")
        result = parse_resume(io.BytesIO(b""))
        for key in ("name", "email", "phone", "skills", "education", "experience", "projects"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()

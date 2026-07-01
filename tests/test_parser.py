"""
Unit tests for Resume Parser module
"""

import pytest
import os
import sys
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import ResumeParser


class TestResumeParser:
    """Test cases for ResumeParser class"""

    @pytest.fixture
    def parser(self):
        """Create a parser instance"""
        return ResumeParser()

    def test_parser_initialization(self, parser):
        """Test parser initialization"""
        assert parser is not None
        assert hasattr(parser, "extract_text")
        assert hasattr(parser, "extract_personal_info")
        assert hasattr(parser, "parse_resume")

    def test_extract_text_file_not_found(self, parser):
        """Test extract_text with non-existent file"""
        with pytest.raises(FileNotFoundError):
            parser.extract_text("non_existent_file.pdf")

    def test_extract_personal_info_email(self, parser):
        """Test email extraction"""
        text = """
        John Doe
        Email: john.doe@email.com
        Phone: 123-456-7890
        """
        info = parser.extract_personal_info(text)
        assert info["email"] == "john.doe@email.com"

    def test_extract_personal_info_phone(self, parser):
        """Test phone number extraction"""
        text = """
        Jane Smith
        Phone: (555) 234-5678
        Email: jane.smith@email.com
        """
        info = parser.extract_personal_info(text)
        assert info["phone"] == "(555) 234-5678"

    def test_extract_personal_info_linkedin(self, parser):
        """Test LinkedIn URL extraction"""
        text = """
        John Doe
        LinkedIn: linkedin.com/in/johndoe
        """
        info = parser.extract_personal_info(text)
        assert "linkedin.com/in/johndoe" in info["linkedin"]

    def test_empty_text_extraction(self, parser):
        """Test extracting from empty text"""
        text = ""
        info = parser.extract_personal_info(text)
        assert info["email"] == ""
        assert info["phone"] == ""
        assert info["linkedin"] == ""

    def test_no_personal_info(self, parser):
        """Test text with no personal information"""
        text = "This is just a plain text with no personal information."
        info = parser.extract_personal_info(text)
        assert all(value == "" for value in info.values())

"""
Resume Parser Module - Extracts text from PDF resumes
"""

import os
import re
from typing import Optional, Dict, List

# Try importing PDF libraries with fallback
try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")


class ResumeParser:
    """
    A class to parse resume PDF files and extract text content
    """

    def __init__(self):
        self.supported_formats = [".pdf"]

    def extract_text_pypdf2(self, file_path: str) -> str:
        """
        Extract text using PyPDF2 library

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text as string
        """
        if not PYPDF2_AVAILABLE:
            print("PyPDF2 is not available")
            return ""

        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text with PyPDF2: {e}")
            return ""

    def extract_text_pdfplumber(self, file_path: str) -> str:
        """
        Extract text using pdfplumber library (better for complex layouts)

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text as string
        """
        if not PDFPLUMBER_AVAILABLE:
            print("pdfplumber is not available")
            return ""

        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {e}")
            return ""

    def extract_text(self, file_path: str) -> str:
        """
        Main method to extract text from PDF

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text as string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try pdfplumber first (better quality), fallback to PyPDF2
        text = self.extract_text_pdfplumber(file_path)

        if not text.strip():
            text = self.extract_text_pypdf2(file_path)

        return text.strip()

    def extract_personal_info(self, text: str) -> Dict[str, str]:
        """
        Extract basic personal information from resume text

        Args:
            text: Resume text

        Returns:
            Dictionary with extracted information
        """
        info = {"name": "", "email": "", "phone": "", "linkedin": "", "location": ""}

        # Extract email
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, text)
        if emails:
            info["email"] = emails[0]

        # Extract phone (US format)
        phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        phones = re.findall(phone_pattern, text)
        if phones:
            info["phone"] = phones[0]

        # Extract LinkedIn
        linkedin_pattern = r"linkedin\.com/in/[A-Za-z0-9-]+"
        linkedin = re.findall(linkedin_pattern, text)
        if linkedin:
            info["linkedin"] = "https://" + linkedin[0]

        return info

    def parse_resume(self, file_path: str) -> Dict[str, str]:
        """
        Parse resume file and extract all information

        Args:
            file_path: Path to PDF file

        Returns:
            Dictionary with parsed information
        """
        text = self.extract_text(file_path)
        personal_info = self.extract_personal_info(text)

        return {
            "file": os.path.basename(file_path),
            "text": text,
            "name": personal_info["name"],
            "email": personal_info["email"],
            "phone": personal_info["phone"],
            "linkedin": personal_info["linkedin"],
        }


# If neither library is available, create a dummy parser
if not PYPDF2_AVAILABLE and not PDFPLUMBER_AVAILABLE:
    print("\n" + "=" * 60)
    print("⚠️  WARNING: No PDF parsing libraries available!")
    print("=" * 60)
    print("Please install at least one of the following:")
    print("  pip install PyPDF2")
    print("  pip install pdfplumber")
    print("=" * 60 + "\n")

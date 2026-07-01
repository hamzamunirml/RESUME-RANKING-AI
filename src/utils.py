"""
Utility Functions for the Resume Screening System
"""

import os
import re
import json
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime


def create_directories(paths: List[str]) -> None:
    """
    Create directories if they don't exist

    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename

    Args:
        filename: Name of the file

    Returns:
        File extension (lowercase)
    """
    return os.path.splitext(filename)[1].lower()


def is_pdf_file(filename: str) -> bool:
    """
    Check if file is a PDF

    Args:
        filename: Name of the file

    Returns:
        True if PDF, False otherwise
    """
    return get_file_extension(filename) == ".pdf"


def is_text_file(filename: str) -> bool:
    """
    Check if file is a text file

    Args:
        filename: Name of the file

    Returns:
        True if text file, False otherwise
    """
    return get_file_extension(filename) in [".txt", ".text"]


def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text

    Args:
        text: Input text

    Returns:
        List of email addresses
    """
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return re.findall(pattern, text)


def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract phone numbers from text

    Args:
        text: Input text

    Returns:
        List of phone numbers
    """
    pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    return re.findall(pattern, text)


def extract_links(text: str) -> List[str]:
    """
    Extract URLs from text

    Args:
        text: Input text

    Returns:
        List of URLs
    """
    pattern = r"(?:http|https):\/\/[A-Za-z0-9\-\.]+\.[A-Za-z]{2,}[\/\w\.\-]*"
    return re.findall(pattern, text)


def save_json(data: Dict, filepath: str) -> None:
    """
    Save data as JSON file

    Args:
        data: Dictionary to save
        filepath: Path to save JSON
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """
    Load JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary from JSON
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_timestamp() -> str:
    """
    Get current timestamp as string

    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_filename(filename: str) -> str:
    """
    Clean filename by removing special characters

    Args:
        filename: Input filename

    Returns:
        Cleaned filename
    """
    return re.sub(r"[^\w\-_. ]", "", filename)


def calculate_experience_years(text: str) -> float:
    """
    Calculate years of experience from text

    Args:
        text: Input text containing experience information

    Returns:
        Years of experience (float)
    """
    # Look for patterns like "X years", "X+ years", "X-Y years"
    patterns = [
        r"(\d+)\+?\s*(?:years?|yrs?)",
        r"(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years?|yrs?)",
    ]

    total_years = 0
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            if isinstance(match, tuple):
                # Range like "3-5 years"
                years = (int(match[0]) + int(match[1])) / 2
            else:
                years = float(match)
            total_years = max(total_years, years)

    return total_years


def analyze_education_level(text: str) -> Dict[str, int]:
    """
    Analyze education levels from text

    Args:
        text: Input text

    Returns:
        Dictionary with education levels and counts
    """
    education_keywords = {
        "phd": ["phd", "doctorate", "doctoral"],
        "masters": ["masters", "ms", "m.sc", "mba", "master"],
        "bachelors": ["bachelors", "bs", "b.sc", "be", "b.tech", "bachelor"],
        "associate": ["associate", "diploma"],
        "high_school": ["high school", "h.s", "school"],
    }

    education_count = {level: 0 for level in education_keywords}

    for level, keywords in education_keywords.items():
        for keyword in keywords:
            count = len(re.findall(r"\b" + keyword + r"\b", text.lower()))
            if count > 0:
                education_count[level] = max(education_count[level], count)

    return education_count


# Test function
def test_utils():
    """Test utility functions"""
    test_text = """
    John Doe has 5 years of experience in software development.
    Contact: john.doe@email.com, phone: 123-456-7890
    Education: MS in Computer Science, B.Tech in IT
    LinkedIn: https://linkedin.com/in/johndoe
    """

    print("Emails:", extract_emails(test_text))
    print("Phone:", extract_phone_numbers(test_text))
    print("Links:", extract_links(test_text))
    print("Experience Years:", calculate_experience_years(test_text))
    print("Education:", analyze_education_level(test_text))


if __name__ == "__main__":
    test_utils()

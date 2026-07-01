"""
Utility functions for Flask app
"""

import os
import sys

sys.path.append("..")

from src.parser import ResumeParser
from src.preprocess import TextPreprocessor
from src.similarity import SimilarityCalculator
from src.ranking import CandidateRanker
import pandas as pd

ALLOWED_EXTENSIONS = {"pdf", "txt"}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_resume(filepath):
    """Process a resume file"""
    parser = ResumeParser()
    preprocessor = TextPreprocessor()

    text = parser.extract_text(filepath)
    cleaned = preprocessor.get_full_preprocessed_text(text)

    return {
        "file": filepath,
        "text": text,
        "cleaned": cleaned,
        "filename": os.path.basename(filepath),
    }


def process_jd(filepath):
    """Process a job description file"""
    preprocessor = TextPreprocessor()

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned = preprocessor.get_full_preprocessed_text(text)

    return {
        "file": filepath,
        "text": text,
        "cleaned": cleaned,
        "filename": os.path.basename(filepath),
    }


def run_screening(resume_files, jd_files):
    """Run the complete screening process"""
    parser = ResumeParser()
    preprocessor = TextPreprocessor()
    calculator = SimilarityCalculator()
    ranker = CandidateRanker()

    # Process resumes
    resume_data = []
    for filepath in resume_files:
        text = parser.extract_text(filepath)
        if text:
            cleaned = preprocessor.get_full_preprocessed_text(text)
            resume_data.append(
                {
                    "file": os.path.basename(filepath),
                    "text": text,
                    "cleaned": cleaned,
                    "name": os.path.basename(filepath)
                    .replace(".pdf", "")
                    .replace("_", " ")
                    .title(),
                }
            )

    # Process job descriptions
    jd_data = []
    for filepath in jd_files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        cleaned = preprocessor.get_full_preprocessed_text(text)
        jd_data.append(
            {
                "file": os.path.basename(filepath),
                "text": text,
                "cleaned": cleaned,
                "title": os.path.basename(filepath)
                .replace(".txt", "")
                .replace("_", " ")
                .title(),
            }
        )

    # Calculate similarity
    resume_texts = [r["cleaned"] for r in resume_data]
    jd_texts = [j["cleaned"] for j in jd_data]

    similarity_matrix = calculator.calculate_similarity_scores(resume_texts, jd_texts)

    # Rank candidates
    names = [r["name"] for r in resume_data]
    files = [r["file"] for r in resume_data]

    results = ranker.rank_candidates(similarity_matrix, names, files, jd_index=0)

    # Save results
    output_file = os.path.join("output", "ranked_candidates.csv")
    results.to_csv(output_file, index=False)

    return results.to_dict("records")

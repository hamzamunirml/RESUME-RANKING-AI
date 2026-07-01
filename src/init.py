"""
AI Resume Screening & Candidate Ranking System
"""

__version__ = "1.0.0"
__author__ = "AI Resume Screening Team"

from .parser import ResumeParser
from .preprocess import TextPreprocessor
from .similarity import SimilarityCalculator
from .ranking import CandidateRanker

"""
Unit tests for Candidate Ranker module
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ranking import CandidateRanker


class TestCandidateRanker:
    """Test cases for CandidateRanker class"""

    @pytest.fixture
    def ranker(self):
        """Create a ranker instance"""
        return CandidateRanker()

    @pytest.fixture
    def sample_similarity_matrix(self):
        """Sample similarity matrix"""
        return np.array(
            [
                [0.92, 0.78, 0.65],
                [0.85, 0.91, 0.70],
                [0.75, 0.82, 0.95],
                [0.60, 0.55, 0.50],
            ]
        )

    @pytest.fixture
    def sample_names(self):
        """Sample candidate names"""
        return ["Alice", "Bob", "Charlie", "Diana"]

    @pytest.fixture
    def sample_files(self):
        """Sample file names"""
        return ["alice.pdf", "bob.pdf", "charlie.pdf", "diana.pdf"]

    def test_initialization(self, ranker):
        """Test ranker initialization"""
        assert ranker is not None
        assert ranker.results is None

    def test_rank_candidates(
        self, ranker, sample_similarity_matrix, sample_names, sample_files
    ):
        """Test candidate ranking"""
        results = ranker.rank_candidates(
            sample_similarity_matrix, sample_names, sample_files, jd_index=0
        )

        assert results is not None
        assert len(results) == len(sample_names)
        assert "Rank" in results.columns
        assert "Candidate" in results.columns
        assert "Similarity_Score" in results.columns
        assert "File" in results.columns

        # Check sorting
        assert results["Similarity_Score"].is_monotonic_decreasing

        # Check rank assignment
        assert results["Rank"].tolist() == list(range(1, len(sample_names) + 1))

    def test_rank_candidates_different_jd(
        self, ranker, sample_similarity_matrix, sample_names, sample_files
    ):
        """Test ranking for different job description"""
        results = ranker.rank_candidates(
            sample_similarity_matrix, sample_names, sample_files, jd_index=1
        )

        assert results is not None
        assert results.iloc[0]["Candidate"] == "Bob"

    def test_rank_candidates_percentage_conversion(
        self, ranker, sample_similarity_matrix, sample_names, sample_files
    ):
        """Test that similarity scores are converted to percentages"""
        results = ranker.rank_candidates(
            sample_similarity_matrix, sample_names, sample_files, jd_index=0
        )

        # Scores should be between 0 and 100
        assert results["Similarity_Score"].min() >= 0
        assert results["Similarity_Score"].max() <= 100

    def test_add_skills_column(
        self, ranker, sample_similarity_matrix, sample_names, sample_files
    ):
        """Test adding skills column"""
        results = ranker.rank_candidates(
            sample_similarity_matrix, sample_names, sample_files, jd_index=0
        )

        skills_dict = {
            "alice.pdf": ["Python", "ML", "NLP"],
            "bob.pdf": ["Java", "Spring", "SQL"],
            "charlie.pdf": ["React", "Node", "MongoDB"],
            "diana.pdf": ["Python", "Django", "AWS"],
        }

        results = ranker.add_skills_column(results, skills_dict)
        assert "Skills" in results.columns

    def test_get_top_candidates(
        self, ranker, sample_similarity_matrix, sample_names, sample_files
    ):
        """Test getting top candidates"""
        ranker.rank_candidates(
            sample_similarity_matrix, sample_names, sample_files, jd_index=0
        )

        top_3 = ranker.get_top_candidates(n=3)
        assert len(top_3) == 3
        assert top_3.iloc[0]["Candidate"] == "Alice"

    def test_generate_summary(
        self, ranker, sample_similarity_matrix, sample_names, sample_files
    ):
        """Test summary generation"""
        ranker.rank_candidates(
            sample_similarity_matrix, sample_names, sample_files, jd_index=0
        )

        summary = ranker.generate_summary()
        assert isinstance(summary, dict)
        assert "total_candidates" in summary
        assert "top_candidate" in summary
        assert "top_score" in summary
        assert summary["total_candidates"] == len(sample_names)

    def test_empty_similarity_matrix(self, ranker):
        """Test empty similarity matrix"""
        matrix = np.array([]).reshape(0, 0)
        results = ranker.rank_candidates(matrix, [], [])
        # Should return empty DataFrame with correct columns
        assert len(results) == 0
        assert list(results.columns) == [
            "Rank",
            "Candidate",
            "Similarity_Score",
            "File",
        ]

    def test_get_top_candidates_no_results(self, ranker):
        """Test get_top_candidates without results"""
        with pytest.raises(ValueError, match="No results available"):
            ranker.get_top_candidates()

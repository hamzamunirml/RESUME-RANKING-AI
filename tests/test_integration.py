"""
Integration tests for the complete system
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import ResumeParser
from src.preprocess import TextPreprocessor
from src.similarity import SimilarityCalculator
from src.ranking import CandidateRanker


class TestIntegration:
    """Integration tests for the complete pipeline"""

    @pytest.fixture
    def sample_data(self):
        """Sample resumes and job descriptions"""
        resumes = [
            """John Doe
            Python Developer with 5 years experience
            Skills: Python, Machine Learning, Deep Learning, NLP, SQL
            Experience: Data Scientist at Google""",
            """Jane Smith
            Full Stack Developer with 4 years experience
            Skills: React, Node.js, MongoDB, Express, AWS
            Experience: Full Stack Developer at Microsoft""",
            """Bob Johnson
            Data Scientist with 3 years experience
            Skills: Python, Machine Learning, Data Analysis, SQL
            Experience: Data Scientist at Amazon""",
        ]

        jds = [
            """Senior Data Scientist
            Skills: Python, Machine Learning, Deep Learning, SQL
            Experience: 3+ years""",
            """Full Stack Developer
            Skills: React, Node.js, MongoDB, AWS
            Experience: 3+ years""",
        ]

        return resumes, jds

    def test_complete_pipeline(self, sample_data):
        """Test the complete screening pipeline"""
        resumes, jds = sample_data

        # Initialize components
        preprocessor = TextPreprocessor()
        calculator = SimilarityCalculator()
        ranker = CandidateRanker()

        # Process resumes
        resume_names = [f"Candidate {i+1}" for i in range(len(resumes))]

        # Preprocess
        cleaned_resumes = [preprocessor.get_full_preprocessed_text(t) for t in resumes]
        cleaned_jds = [preprocessor.get_full_preprocessed_text(t) for t in jds]

        # Calculate similarity
        similarity_matrix = calculator.calculate_similarity_scores(
            cleaned_resumes, cleaned_jds
        )

        # Rank candidates for first JD
        results = ranker.rank_candidates(
            similarity_matrix,
            resume_names,
            [f"resume_{i+1}.pdf" for i in range(len(resumes))],
            jd_index=0,
        )

        # Verify results
        assert len(results) == len(resumes)
        assert "Rank" in results.columns
        assert "Candidate" in results.columns
        assert "Similarity_Score" in results.columns

        # The candidate with "Data Scientist" should rank higher for data scientist JD
        assert results.iloc[0]["Candidate"] in ["Candidate 1", "Candidate 3"]

    def test_end_to_end_ranking(self, sample_data):
        """Test end-to-end ranking"""
        resumes, jds = sample_data

        preprocessor = TextPreprocessor()
        calculator = SimilarityCalculator()
        ranker = CandidateRanker()

        # Process data
        cleaned_resumes = [preprocessor.get_full_preprocessed_text(t) for t in resumes]
        cleaned_jds = [preprocessor.get_full_preprocessed_text(t) for t in jds]

        similarity_matrix = calculator.calculate_similarity_scores(
            cleaned_resumes, cleaned_jds
        )

        # Test different JDs
        for jd_idx in range(len(jds)):
            results = ranker.rank_candidates(
                similarity_matrix,
                [f"Resume {i+1}" for i in range(len(resumes))],
                [f"resume_{i+1}.pdf" for i in range(len(resumes))],
                jd_index=jd_idx,
            )

            # Verify each run produces valid results
            assert len(results) == len(resumes)
            assert results["Similarity_Score"].max() <= 100
            assert results["Similarity_Score"].min() >= 0

    def test_full_pipeline_with_skills(self, sample_data):
        """Test full pipeline including skill extraction"""
        resumes, jds = sample_data

        preprocessor = TextPreprocessor()
        calculator = SimilarityCalculator()
        ranker = CandidateRanker()

        # Process
        cleaned_resumes = [preprocessor.get_full_preprocessed_text(t) for t in resumes]
        cleaned_jds = [preprocessor.get_full_preprocessed_text(t) for t in jds]

        similarity_matrix = calculator.calculate_similarity_scores(
            cleaned_resumes, cleaned_jds
        )

        results = ranker.rank_candidates(
            similarity_matrix,
            ["Candidate 1", "Candidate 2", "Candidate 3"],
            ["resume1.pdf", "resume2.pdf", "resume3.pdf"],
            jd_index=0,
        )

        # Add skills manually for test
        skills = {
            "resume1.pdf": ["Python", "ML", "NLP"],
            "resume2.pdf": ["React", "Node", "AWS"],
            "resume3.pdf": ["Python", "SQL", "Data Analysis"],
        }

        results = ranker.add_skills_column(results, skills)

        assert "Skills" in results.columns
        # Check that results are sorted by score
        assert results["Similarity_Score"].is_monotonic_decreasing

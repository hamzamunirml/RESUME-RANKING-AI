"""
Unit tests for Similarity Calculator module
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.similarity import SimilarityCalculator


class TestSimilarityCalculator:
    """Test cases for SimilarityCalculator class"""

    @pytest.fixture
    def calculator(self):
        """Create a similarity calculator instance"""
        return SimilarityCalculator()

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            "Python developer with machine learning experience",
            "Data scientist with deep learning skills",
            "Full stack developer with React and Node.js",
        ]

    def test_initialization(self, calculator):
        """Test calculator initialization"""
        assert calculator is not None
        assert calculator.max_features == 5000
        assert calculator.ngram_range == (1, 2)
        assert calculator.vectorizer is None

    def test_fit_vectorizer(self, calculator, sample_documents):
        """Test vectorizer fitting"""
        calculator.fit_vectorizer(sample_documents)
        assert calculator.vectorizer is not None
        assert calculator.vectors is not None
        assert calculator.feature_names is not None
        assert len(calculator.feature_names) > 0

    def test_get_vectors(self, calculator, sample_documents):
        """Test getting vectors"""
        calculator.fit_vectorizer(sample_documents)
        vectors = calculator.get_vectors(sample_documents)
        assert vectors is not None
        assert vectors.shape[0] == len(sample_documents)

    def test_get_vectors_without_fit(self, calculator, sample_documents):
        """Test getting vectors without fitting"""
        with pytest.raises(ValueError, match="Vectorizer not fitted"):
            calculator.get_vectors(sample_documents)

    def test_calculate_similarity(self, calculator, sample_documents):
        """Test similarity calculation"""
        calculator.fit_vectorizer(sample_documents)
        vectors = calculator.get_vectors(sample_documents)

        # Same documents should have high similarity
        similarity = calculator.calculate_similarity(vectors, vectors)
        assert similarity.shape == (len(sample_documents), len(sample_documents))

        # Diagonal should be 1.0 (self-similarity)
        for i in range(len(sample_documents)):
            assert similarity[i][i] == pytest.approx(1.0, abs=0.01)

    def test_calculate_similarity_scores(self, calculator):
        """Test full similarity scoring pipeline"""
        resumes = [
            "Python developer with ML experience",
            "Data scientist with deep learning",
        ]
        jds = ["Need Python developer with ML skills", "Looking for data scientist"]

        similarity_matrix = calculator.calculate_similarity_scores(resumes, jds)
        assert similarity_matrix.shape == (len(resumes), len(jds))

    def test_get_matching_features(self, calculator):
        """Test getting matching features"""
        resume = "Python developer with machine learning"
        jd = "Need Python developer with ML skills"

        matches = calculator.get_matching_features(resume, jd, top_n=5)
        assert len(matches) > 0

    def test_identical_documents(self, calculator):
        """Test identical documents"""
        docs = ["Python developer", "Python developer"]
        calculator.fit_vectorizer(docs)
        vectors = calculator.get_vectors(docs)
        similarity = calculator.calculate_similarity(vectors, vectors)
        assert similarity[0][1] == pytest.approx(1.0, abs=0.01)

    def test_completely_different_documents(self, calculator):
        """Test completely different documents"""
        docs = ["Python developer", "Baking recipes"]
        calculator.fit_vectorizer(docs)
        vectors = calculator.get_vectors(docs)
        similarity = calculator.calculate_similarity(vectors, vectors)
        # Should have low similarity
        assert similarity[0][1] < 0.3

"""
Similarity Calculation Module - Computes similarity between resumes and job descriptions
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
import pandas as pd


class SimilarityCalculator:
    """
    A class to calculate similarity scores between text documents
    """

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the similarity calculator

        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to consider
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.vectors = None
        self.feature_names = None

    def fit_vectorizer(self, documents: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on documents

        Args:
            documents: List of document texts
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            lowercase=True,
        )
        self.vectors = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def get_vectors(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF vectors

        Args:
            documents: List of document texts

        Returns:
            TF-IDF vectors
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")

        return self.vectorizer.transform(documents)

    def calculate_similarity(
        self, doc1_vectors: np.ndarray, doc2_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between two sets of vectors

        Args:
            doc1_vectors: Vectors for first set of documents
            doc2_vectors: Vectors for second set of documents

        Returns:
            Similarity matrix
        """
        return cosine_similarity(doc1_vectors, doc2_vectors)

    def calculate_similarity_scores(
        self, resume_texts: List[str], jd_texts: List[str]
    ) -> np.ndarray:
        """
        Calculate similarity scores between resumes and job descriptions

        Args:
            resume_texts: List of resume texts
            jd_texts: List of job description texts

        Returns:
            Similarity matrix (n_resumes x n_jds)
        """
        # Combine all documents for vectorization
        all_texts = resume_texts + jd_texts

        # Fit vectorizer
        self.fit_vectorizer(all_texts)

        # Get vectors
        resume_vectors = self.get_vectors(resume_texts)
        jd_vectors = self.get_vectors(jd_texts)

        # Calculate similarity
        similarity_matrix = self.calculate_similarity(resume_vectors, jd_vectors)

        return similarity_matrix

    def get_matching_features(
        self, resume_text: str, jd_text: str, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get matching features between a resume and job description

        Args:
            resume_text: Resume text
            jd_text: Job description text
            top_n: Number of top features to return

        Returns:
            DataFrame with feature matches
        """
        if self.vectorizer is None:
            # Fit vectorizer with these two documents
            self.fit_vectorizer([resume_text, jd_text])

        # Get vectors
        resume_vector = self.get_vectors([resume_text]).toarray().flatten()
        jd_vector = self.get_vectors([jd_text]).toarray().flatten()

        # Get feature names
        feature_names = self.feature_names

        # Create DataFrame
        features_df = pd.DataFrame(
            {
                "feature": feature_names,
                "resume_score": resume_vector,
                "jd_score": jd_vector,
            }
        )

        # Get features that appear in both
        features_df["both"] = (features_df["resume_score"] > 0) & (
            features_df["jd_score"] > 0
        )
        features_df["similarity_contribution"] = (
            features_df["resume_score"] * features_df["jd_score"]
        )

        # Sort by contribution
        matching_features = features_df[features_df["both"]].sort_values(
            "similarity_contribution", ascending=False
        )

        return matching_features.head(top_n)

    def get_important_features(self, text: str, top_n: int = 20) -> List[str]:
        """
        Get important features/terms from a text

        Args:
            text: Input text
            top_n: Number of top features to return

        Returns:
            List of important features
        """
        if self.vectorizer is None:
            self.fit_vectorizer([text])

        vector = self.get_vectors([text]).toarray().flatten()
        feature_names = self.feature_names

        # Get features with non-zero scores
        features_df = pd.DataFrame({"feature": feature_names, "score": vector})

        important = features_df[features_df["score"] > 0].sort_values(
            "score", ascending=False
        )
        return important.head(top_n)["feature"].tolist()


# Test function
def test_similarity():
    """Test the similarity calculator"""
    calculator = SimilarityCalculator()

    # Sample texts
    resume_text = (
        "Experienced Python developer with skills in machine learning and data science"
    )
    jd_text = (
        "We are looking for a Python developer with experience in machine learning"
    )

    # Calculate similarity
    similarity_matrix = calculator.calculate_similarity_scores([resume_text], [jd_text])

    print(f"Similarity Score: {similarity_matrix[0][0]:.4f}")

    # Get matching features
    matches = calculator.get_matching_features(resume_text, jd_text)
    print("\nMatching Features:")
    print(matches)


if __name__ == "__main__":
    test_similarity()

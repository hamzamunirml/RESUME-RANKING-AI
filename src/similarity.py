"""
Similarity Calculation Module - Computes similarity between resumes and job descriptions

FIXED VERSION:
Previously, calculate_similarity_scores() called fit_vectorizer() fresh on
just the 2-3 documents being compared. With such a tiny corpus, TF-IDF's
IDF weighting is statistically meaningless, which produced artificially
low similarity scores (e.g. ~14% even when skills clearly matched).

Now the vectorizer is trained ONCE on a large reference corpus
(see build_vectorizer.py) and loaded here. Only .transform() is called
at inference time, never .fit() again.
"""

import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
import pandas as pd

DEFAULT_VECTORIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "tfidf_vectorizer.pkl"
)


class SimilarityCalculator:
    """
    A class to calculate similarity scores between text documents,
    using a pretrained TF-IDF vectorizer.
    """

    def __init__(
        self,
        vectorizer_path: str = DEFAULT_VECTORIZER_PATH,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 1),
    ):
        """
        Args:
            vectorizer_path: path to a pretrained, pickled TfidfVectorizer
                              (created by build_vectorizer.py). If the file
                              doesn't exist yet, falls back to fitting on
                              whatever documents are passed in at runtime
                              (old behavior) with a printed warning.
            max_features / ngram_range: only used for the fallback fit.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer_path = vectorizer_path
        self.vectorizer = None
        self.feature_names = None
        self._load_pretrained_vectorizer()

    def _load_pretrained_vectorizer(self) -> None:
        if os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.feature_names = self.vectorizer.get_feature_names_out()
        else:
            print(
                f"[SimilarityCalculator] WARNING: no pretrained vectorizer found at "
                f"'{self.vectorizer_path}'. Run build_vectorizer.py first for accurate "
                f"scores. Falling back to fitting on runtime documents (less reliable)."
            )

    def fit_vectorizer(self, documents: List[str]) -> None:
        """
        Fallback only: fits a new vectorizer on the given documents.
        Prefer using a pretrained vectorizer (build_vectorizer.py) instead —
        fitting on a tiny document set makes IDF weights unreliable and
        similarity scores artificially low.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            lowercase=True,
        )
        self.vectorizer.fit(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def get_vectors(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF vectors using the (pretrained) vectorizer.
        """
        if self.vectorizer is None:
            raise ValueError(
                "Vectorizer not available. Run build_vectorizer.py first, "
                "or call fit_vectorizer() manually."
            )
        return self.vectorizer.transform(documents)

    def calculate_similarity(
        self, doc1_vectors: np.ndarray, doc2_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between two sets of vectors
        """
        return cosine_similarity(doc1_vectors, doc2_vectors)

    def calculate_similarity_scores(
        self, resume_texts: List[str], jd_texts: List[str]
    ) -> np.ndarray:
        """
        Calculate similarity scores between resumes and job descriptions.

        Uses the pretrained vectorizer (loaded in __init__) — does NOT
        refit on resume_texts/jd_texts. This is the key fix: scores are
        now stable and meaningful regardless of how many documents are
        being compared in a single call (even just 1 resume vs 1 JD).
        """
        if self.vectorizer is None:
            # Fallback path only — see warning in _load_pretrained_vectorizer
            self.fit_vectorizer(resume_texts + jd_texts)

        resume_vectors = self.get_vectors(resume_texts)
        jd_vectors = self.get_vectors(jd_texts)

        return self.calculate_similarity(resume_vectors, jd_vectors)

    def get_matching_features(
        self, resume_text: str, jd_text: str, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get matching features between a resume and job description
        """
        if self.vectorizer is None:
            self.fit_vectorizer([resume_text, jd_text])

        resume_vector = self.get_vectors([resume_text]).toarray().flatten()
        jd_vector = self.get_vectors([jd_text]).toarray().flatten()

        feature_names = self.feature_names

        features_df = pd.DataFrame(
            {
                "feature": feature_names,
                "resume_score": resume_vector,
                "jd_score": jd_vector,
            }
        )

        features_df["both"] = (features_df["resume_score"] > 0) & (
            features_df["jd_score"] > 0
        )
        features_df["similarity_contribution"] = (
            features_df["resume_score"] * features_df["jd_score"]
        )

        matching_features = features_df[features_df["both"]].sort_values(
            "similarity_contribution", ascending=False
        )

        return matching_features.head(top_n)

    def get_important_features(self, text: str, top_n: int = 20) -> List[str]:
        """
        Get important features/terms from a text
        """
        if self.vectorizer is None:
            self.fit_vectorizer([text])

        vector = self.get_vectors([text]).toarray().flatten()
        feature_names = self.feature_names

        features_df = pd.DataFrame({"feature": feature_names, "score": vector})

        important = features_df[features_df["score"] > 0].sort_values(
            "score", ascending=False
        )
        return important.head(top_n)["feature"].tolist()


# Test function
def test_similarity():
    """Test the similarity calculator"""
    calculator = SimilarityCalculator()

    resume_text = (
        "Experienced Python developer with skills in machine learning and data science"
    )
    jd_text = (
        "We are looking for a Python developer with experience in machine learning"
    )

    similarity_matrix = calculator.calculate_similarity_scores([resume_text], [jd_text])
    print(f"Similarity Score: {similarity_matrix[0][0]:.4f}")

    matches = calculator.get_matching_features(resume_text, jd_text)
    print("\nMatching Features:")
    print(matches)


if __name__ == "__main__":
    test_similarity()

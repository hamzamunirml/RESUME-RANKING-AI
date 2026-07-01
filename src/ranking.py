"""
Candidate Ranking Module - Ranks candidates based on similarity scores
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CandidateRank:
    """Data class for candidate ranking information"""

    rank: int
    name: str
    file: str
    similarity_score: float
    skills: List[str]
    missing_skills: List[str]
    experience_score: Optional[float] = None
    education_score: Optional[float] = None


class CandidateRanker:
    """
    A class to rank candidates based on resume-job description similarity
    """

    def __init__(self):
        self.results = None

    def rank_candidates(
        self,
        similarity_matrix: np.ndarray,
        resume_names: List[str],
        resume_files: List[str],
        jd_index: int = 0,
    ) -> pd.DataFrame:
        """
        Rank candidates for a specific job description

        Args:
            similarity_matrix: Matrix of similarity scores
            resume_names: List of candidate names
            resume_files: List of resume file names
            jd_index: Index of job description to rank for

        Returns:
            DataFrame with ranked candidates
        """
        # ✅ FIX: Handle empty matrix
        if similarity_matrix.shape[0] == 0 or similarity_matrix.shape[1] == 0:
            return pd.DataFrame(
                columns=["Rank", "Candidate", "Similarity_Score", "File"]
            )

        # Get scores for specific JD
        scores = similarity_matrix[:, jd_index]

        # Create results DataFrame
        results = pd.DataFrame(
            {
                "Candidate": resume_names,
                "File": resume_files,
                "Similarity_Score": scores * 100,  # Convert to percentage
            }
        )

        # Sort by score descending
        results = results.sort_values("Similarity_Score", ascending=False).reset_index(
            drop=True
        )

        # Add rank
        results["Rank"] = range(1, len(results) + 1)

        # Reorder columns
        results = results[["Rank", "Candidate", "Similarity_Score", "File"]]

        self.results = results
        return results

    def add_skills_column(
        self, results: pd.DataFrame, resume_skills: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Add skills to the results DataFrame

        Args:
            results: Results DataFrame
            resume_skills: Dictionary mapping file names to skill lists

        Returns:
            Updated DataFrame with skills
        """
        results["Skills"] = results["File"].apply(
            lambda f: ", ".join(resume_skills.get(f, [])[:5])
        )
        return results

    def add_missing_skills(
        self,
        results: pd.DataFrame,
        resume_skills: Dict[str, List[str]],
        jd_skills: List[str],
    ) -> pd.DataFrame:
        """
        Add missing skills to the results DataFrame

        Args:
            results: Results DataFrame
            resume_skills: Dictionary mapping file names to skill lists
            jd_skills: List of required skills from job description

        Returns:
            Updated DataFrame with missing skills
        """

        def get_missing_skills(file):
            candidate_skills = set(resume_skills.get(file, []))
            required_skills = set(jd_skills)
            missing = required_skills - candidate_skills
            return ", ".join(list(missing)[:5]) if missing else "None"

        results["Missing_Skills"] = results["File"].apply(get_missing_skills)
        return results

    def add_scoring_metrics(
        self,
        results: pd.DataFrame,
        experience_scores: Dict[str, float],
        education_scores: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Add additional scoring metrics

        Args:
            results: Results DataFrame
            experience_scores: Dictionary mapping file names to experience scores
            education_scores: Dictionary mapping file names to education scores

        Returns:
            Updated DataFrame with scoring metrics
        """
        results["Experience_Score"] = results["File"].apply(
            lambda f: experience_scores.get(f, 0)
        )
        results["Education_Score"] = results["File"].apply(
            lambda f: education_scores.get(f, 0)
        )

        # Calculate composite score (weighted average)
        results["Composite_Score"] = (
            results["Similarity_Score"] * 0.4
            + results["Experience_Score"] * 0.3
            + results["Education_Score"] * 0.3
        )

        # Re-rank by composite score
        results = results.sort_values("Composite_Score", ascending=False).reset_index(
            drop=True
        )
        results["Rank"] = range(1, len(results) + 1)

        return results

    def get_top_candidates(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N candidates

        Args:
            n: Number of candidates to return

        Returns:
            DataFrame with top candidates
        """
        if self.results is None:
            raise ValueError("No results available. Call rank_candidates first.")

        return self.results.head(n)

    def export_to_csv(self, filepath: str) -> None:
        """
        Export results to CSV

        Args:
            filepath: Path to save CSV file
        """
        if self.results is None:
            raise ValueError("No results available. Call rank_candidates first.")

        self.results.to_csv(filepath, index=False)

    def generate_summary(self) -> Dict:
        """
        Generate summary statistics about the ranking

        Returns:
            Dictionary with summary statistics
        """
        if self.results is None:
            raise ValueError("No results available. Call rank_candidates first.")

        summary = {
            "total_candidates": len(self.results),
            "top_candidate": self.results.iloc[0]["Candidate"],
            "top_score": self.results.iloc[0]["Similarity_Score"],
            "average_score": self.results["Similarity_Score"].mean(),
            "median_score": self.results["Similarity_Score"].median(),
            "std_score": self.results["Similarity_Score"].std(),
            "min_score": self.results["Similarity_Score"].min(),
            "max_score": self.results["Similarity_Score"].max(),
        }

        return summary

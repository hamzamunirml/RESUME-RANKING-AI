#!/usr/bin/env python
"""
AI Resume Screening & Candidate Ranking System
Main application entry point
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import project modules
from src.parser import ResumeParser
from src.preprocess import TextPreprocessor
from src.similarity import SimilarityCalculator
from src.ranking import CandidateRanker
from src.utils import create_directories, get_timestamp


class ResumeScreeningSystem:
    """
    Main class for the Resume Screening System
    """

    def __init__(self, data_dir: str = "data/", output_dir: str = "output/"):
        """
        Initialize the screening system

        Args:
            data_dir: Directory containing data
            output_dir: Directory for output
        """
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Setup directories
        self.resume_dir = os.path.join(data_dir, "resumes")
        self.jd_dir = os.path.join(data_dir, "job_descriptions")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.charts_dir = os.path.join(output_dir, "charts")

        # Create directories
        create_directories(
            [
                self.resume_dir,
                self.jd_dir,
                self.processed_dir,
                self.output_dir,
                self.charts_dir,
                "images/",
            ]
        )

        # Initialize components
        self.parser = ResumeParser()
        self.preprocessor = TextPreprocessor()
        self.similarity = SimilarityCalculator()
        self.ranker = CandidateRanker()

        self.resume_data = []
        self.jd_data = []
        self.results = None

    def load_resumes(self) -> None:
        """Load and parse all resumes"""
        print("\n" + "=" * 60)
        print("STEP 1: Loading Resumes")
        print("=" * 60)

        resume_files = [f for f in os.listdir(self.resume_dir) if f.endswith(".pdf")]

        if not resume_files:
            print("No PDF resumes found in data/resumes/")
            print("Please add resume PDF files to the data/resumes/ directory.")
            return

        print(f"Found {len(resume_files)} resume(s)")

        for resume_file in resume_files:
            filepath = os.path.join(self.resume_dir, resume_file)
            print(f"  - Processing: {resume_file}")

            try:
                # Extract text
                text = self.parser.extract_text(filepath)
                if not text:
                    print(f"    ⚠️  No text extracted from {resume_file}")
                    continue

                # Clean and preprocess
                cleaned = self.preprocessor.clean_text(text)
                tokens = self.preprocessor.preprocess(cleaned)

                self.resume_data.append(
                    {
                        "file": resume_file,
                        "name": resume_file.replace(".pdf", "")
                        .replace("_", " ")
                        .title(),
                        "original_text": text,
                        "cleaned_text": cleaned,
                        "tokens": tokens,
                        "full_text": " ".join(tokens),
                    }
                )
                print(f"    ✓ {resume_file} loaded ({len(tokens)} tokens)")

            except Exception as e:
                print(f"    ✗ Error processing {resume_file}: {str(e)}")

        print(f"\n✓ Successfully loaded {len(self.resume_data)} resumes")

    def load_job_descriptions(self) -> None:
        """Load all job descriptions"""
        print("\n" + "=" * 60)
        print("STEP 2: Loading Job Descriptions")
        print("=" * 60)

        jd_files = [f for f in os.listdir(self.jd_dir) if f.endswith(".txt")]

        if not jd_files:
            print("No job descriptions found in data/job_descriptions/")
            print("Please add .txt files to data/job_descriptions/")
            return

        print(f"Found {len(jd_files)} job description(s)")

        for jd_file in jd_files:
            filepath = os.path.join(self.jd_dir, jd_file)
            print(f"  - Processing: {jd_file}")

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

                if not text:
                    print(f"    ⚠️  No content in {jd_file}")
                    continue

                # Clean and preprocess
                cleaned = self.preprocessor.clean_text(text)
                tokens = self.preprocessor.preprocess(cleaned)

                self.jd_data.append(
                    {
                        "file": jd_file,
                        "title": jd_file.replace("_", " ").replace(".txt", "").title(),
                        "original_text": text,
                        "cleaned_text": cleaned,
                        "tokens": tokens,
                        "full_text": " ".join(tokens),
                    }
                )
                print(f"    ✓ {jd_file} loaded ({len(tokens)} tokens)")

            except Exception as e:
                print(f"    ✗ Error processing {jd_file}: {str(e)}")

        print(f"\n✓ Successfully loaded {len(self.jd_data)} job descriptions")

    def calculate_similarity(self) -> None:
        """Calculate similarity between resumes and job descriptions"""
        print("\n" + "=" * 60)
        print("STEP 3: Calculating Similarity Scores")
        print("=" * 60)

        if not self.resume_data or not self.jd_data:
            print("⚠️  Missing resume or JD data. Cannot calculate similarity.")
            return

        resume_texts = [r["full_text"] for r in self.resume_data]
        jd_texts = [j["full_text"] for j in self.jd_data]

        print(
            f"Computing similarity between {len(resume_texts)} resumes and {len(jd_texts)} job descriptions..."
        )

        # Calculate similarity matrix
        self.similarity_matrix = self.similarity.calculate_similarity_scores(
            resume_texts, jd_texts
        )

        print(f"✓ Similarity matrix shape: {self.similarity_matrix.shape}")

    def rank_candidates(self) -> None:
        """Rank candidates for each job description"""
        print("\n" + "=" * 60)
        print("STEP 4: Ranking Candidates")
        print("=" * 60)

        if self.similarity_matrix is None:
            print("⚠️  No similarity matrix available. Calculate similarity first.")
            return

        resume_names = [r["name"] for r in self.resume_data]
        resume_files = [r["file"] for r in self.resume_data]

        all_results = []

        for jd_idx, jd in enumerate(self.jd_data):
            print(f"\nRanking for: {jd['title']}")

            # Rank candidates for this JD
            results = self.ranker.rank_candidates(
                self.similarity_matrix, resume_names, resume_files, jd_index=jd_idx
            )

            # Add JD information
            results["Job_Description"] = jd["title"]

            all_results.append(results)

            # Show top 5
            print(f"  Top 5 candidates for {jd['title']}:")
            for _, row in results.head(5).iterrows():
                print(
                    f"    #{row['Rank']}: {row['Candidate']} - {row['Similarity_Score']:.1f}%"
                )

        # Combine all results
        if all_results:
            self.all_results = pd.concat(all_results, ignore_index=True)
            print(f"\n✓ Ranking completed for {len(all_results)} candidates")

    def generate_skills_analysis(self) -> None:
        """Generate skills analysis for candidates"""
        print("\n" + "=" * 60)
        print("STEP 5: Skills Analysis")
        print("=" * 60)

        # Define skill keywords
        skill_keywords = [
            "python",
            "java",
            "javascript",
            "react",
            "angular",
            "vue",
            "node",
            "django",
            "flask",
            "spring",
            "tensorflow",
            "pytorch",
            "machine learning",
            "deep learning",
            "nlp",
            "computer vision",
            "data science",
            "sql",
            "mysql",
            "postgresql",
            "mongodb",
            "redis",
            "aws",
            "azure",
            "docker",
            "kubernetes",
            "git",
            "linux",
            "html",
            "css",
            "rest api",
            "graphql",
            "c++",
            "c#",
            "ruby",
            "php",
            "swift",
            "kotlin",
            "go",
            "typescript",
            "javascript",
        ]

        def detect_skills(text):
            """Detect skills in text"""
            text_lower = text.lower()
            found = []
            for skill in skill_keywords:
                if skill in text_lower:
                    found.append(skill)
            return found

        # Add skills to results
        if hasattr(self, "all_results"):
            skill_dict = {
                r["file"]: detect_skills(r["original_text"]) for r in self.resume_data
            }

            self.all_results["Skills"] = self.all_results["File"].apply(
                lambda f: ", ".join(skill_dict.get(f, [])[:5])
            )

            self.all_results["Skill_Count"] = self.all_results["File"].apply(
                lambda f: len(skill_dict.get(f, []))
            )

            print(f"✓ Skills analysis completed for {len(self.all_results)} candidates")
        else:
            print("⚠️  No results available. Rank candidates first.")

    def export_results(self) -> None:
        """Export results to CSV"""
        print("\n" + "=" * 60)
        print("STEP 6: Exporting Results")
        print("=" * 60)

        if not hasattr(self, "all_results") or self.all_results is None:
            print("⚠️  No results to export. Rank candidates first.")
            return

        # Export detailed results
        output_file = os.path.join(self.output_dir, "ranked_candidates.csv")
        self.all_results.to_csv(output_file, index=False)
        print(f"✓ Detailed results saved to: {output_file}")

        # Export summary for each JD
        for jd_name in self.all_results["Job_Description"].unique():
            jd_results = self.all_results[
                self.all_results["Job_Description"] == jd_name
            ]
            summary_file = os.path.join(
                self.output_dir, f"summary_{jd_name.replace(' ', '_')}.csv"
            )
            jd_results.to_csv(summary_file, index=False)
            print(f"✓ Summary saved to: {summary_file}")

    def print_summary(self) -> None:
        """Print summary statistics"""
        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)

        print(f"\nTotal Resumes Processed: {len(self.resume_data)}")
        print(f"Total Job Descriptions: {len(self.jd_data)}")

        if hasattr(self, "all_results") and self.all_results is not None:
            print(f"\nTotal Rankings Generated: {len(self.all_results)}")

            # For each JD
            for jd_name in self.all_results["Job_Description"].unique():
                jd_results = self.all_results[
                    self.all_results["Job_Description"] == jd_name
                ]
                print(f"\n{jd_name}:")
                print(f"  - Candidates ranked: {len(jd_results)}")
                print(f"  - Top score: {jd_results['Similarity_Score'].max():.1f}%")
                print(
                    f"  - Average score: {jd_results['Similarity_Score'].mean():.1f}%"
                )
                print(
                    f"  - Top candidate: {jd_results.iloc[0]['Candidate']} ({jd_results.iloc[0]['Similarity_Score']:.1f}%)"
                )

    def run(self) -> None:
        """Run the complete screening pipeline"""
        print("\n" + "=" * 60)
        print("🤖 AI RESUME SCREENING & CANDIDATE RANKING SYSTEM")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Check for data
        if not os.path.exists(self.resume_dir):
            print(f"\n⚠️  Resume directory not found: {self.resume_dir}")
            print("Please create the directory and add resume PDF files.")
            return

        if not os.path.exists(self.jd_dir):
            print(f"\n⚠️  Job description directory not found: {self.jd_dir}")
            print("Please create the directory and add job description TXT files.")
            return

        try:
            # Run all steps
            self.load_resumes()
            self.load_job_descriptions()

            if self.resume_data and self.jd_data:
                self.calculate_similarity()
                self.rank_candidates()
                self.generate_skills_analysis()
                self.export_results()
                self.print_summary()
            else:
                print("\n⚠️  Insufficient data for processing.")
                print("Please ensure:")
                print("  - data/resumes/ contains PDF files")
                print("  - data/job_descriptions/ contains TXT files")

            print("\n" + "=" * 60)
            print("✅ PROCESSING COMPLETE")
            print("=" * 60)
            print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Output directory: {self.output_dir}")

        except Exception as e:
            print(f"\n❌ Error during processing: {str(e)}")
            import traceback

            traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI Resume Screening & Candidate Ranking System"
    )
    parser.add_argument(
        "--data-dir", default="data/", help="Directory containing data (default: data/)"
    )
    parser.add_argument(
        "--output-dir",
        default="output/",
        help="Directory for output files (default: output/)",
    )

    args = parser.parse_args()

    # Run the system
    system = ResumeScreeningSystem(data_dir=args.data_dir, output_dir=args.output_dir)
    system.run()


if __name__ == "__main__":
    main()

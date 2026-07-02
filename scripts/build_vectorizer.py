"""
build_vectorizer.py
--------------------
Trains a TF-IDF vectorizer ONCE on a large reference corpus and saves it
to disk. The Streamlit app / notebooks should LOAD this saved vectorizer
and only call .transform() on new resume/JD text — never .fit() again.

This fixes the core bug: fitting TF-IDF fresh on just 2 documents
(1 resume + 1 JD) makes IDF statistics meaningless and produces
artificially low similarity scores.

Run this once (or whenever you add more reference data):
    python build_vectorizer.py
"""

import os
import glob
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import TextPreprocessor


# ---------------------------------------------------------------------
# 1. Reference corpus
# ---------------------------------------------------------------------
# Ideally: use ALL resumes + ALL job descriptions you have in
# ../data/resumes/ and ../data/job_descriptions/ (the more, the better —
# aim for 50+ documents so IDF weights are statistically meaningful).
#
# If you don't have enough real data yet, this script also mixes in a
# small set of generic role-based reference texts covering different
# job families, so common words (python, experience, team, etc.) get
# realistic IDF weights instead of being treated as rare/important.
# ---------------------------------------------------------------------

GENERIC_REFERENCE_TEXTS = [
    "senior software engineer python java react node experience building "
    "scalable web applications microservices architecture agile team",

    "data analyst sql tableau powerbi excel reporting dashboards business "
    "intelligence data visualization stakeholder communication",

    "frontend developer html css javascript react vue angular responsive "
    "design ui ux figma component library",

    "devops engineer docker kubernetes aws azure ci cd pipeline automation "
    "infrastructure monitoring linux terraform",

    "marketing manager social media campaign strategy content branding seo "
    "google analytics email marketing budget",

    "accountant financial reporting budgeting audit tax compliance excel "
    "quickbooks reconciliation invoicing",

    "project manager agile scrum stakeholder communication timeline planning "
    "budget risk management jira confluence",

    "mobile developer android ios flutter kotlin swift app development "
    "firebase rest api push notifications",

    "machine learning engineer python scikit learn tensorflow pytorch nlp "
    "computer vision model deployment mlops data pipeline",

    "backend developer python django flask fastapi rest api database "
    "postgresql redis microservices docker",

    "product manager roadmap prioritization user research stakeholder "
    "management analytics a b testing agile",

    "hr recruiter talent acquisition interviewing onboarding employee "
    "relations payroll benefits compliance",

    "network engineer cisco routing switching firewall vpn security "
    "troubleshooting infrastructure monitoring",

    "graphic designer adobe photoshop illustrator branding typography "
    "layout print digital design creative",

    "sales executive lead generation crm negotiation client relationship "
    "quota pipeline management closing deals",

    "qa engineer manual testing automation selenium test cases bug tracking "
    "regression testing quality assurance",

    "cloud engineer aws azure gcp terraform infrastructure as code "
    "networking security compliance scalability",

    "content writer seo copywriting editing blog articles social media "
    "content strategy research audience engagement",

    "cybersecurity analyst threat detection incident response penetration "
    "testing vulnerability assessment compliance",

    "database administrator sql server mysql postgresql backup recovery "
    "performance tuning replication high availability",
]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # assumes this script lives in scripts/


def load_local_documents():
    """Load any real resumes/JDs from data/ if present."""
    docs = []
    resume_dir = os.path.join(PROJECT_ROOT, "data", "resumes")
    jd_dir = os.path.join(PROJECT_ROOT, "data", "job_descriptions")

    if os.path.isdir(resume_dir):
        for path in glob.glob(os.path.join(resume_dir, "*.txt")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())

    if os.path.isdir(jd_dir):
        for path in glob.glob(os.path.join(jd_dir, "*.txt")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())

    return docs


def main():
    preprocessor = TextPreprocessor()

    raw_docs = GENERIC_REFERENCE_TEXTS + load_local_documents()
    print(f"Total reference documents: {len(raw_docs)}")

    cleaned_docs = [preprocessor.get_full_preprocessed_text(d) for d in raw_docs]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),   # unigrams only — bigrams are too sparse for small corpora
        stop_words="english",
        lowercase=True,
        min_df=1,
    )
    vectorizer.fit(cleaned_docs)

    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, out_path)

    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"Vectorizer saved to: {out_path}")
    print("\nDone. Now update similarity.py to LOAD this vectorizer instead of fitting a new one each time.")


if __name__ == "__main__":
    main()

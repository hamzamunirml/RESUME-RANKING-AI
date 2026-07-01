"""
Application Routes
"""

from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    send_file,
    flash,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json
from datetime import datetime
import sys

sys.path.append("..")

from app.utils import process_resume, process_jd, run_screening, allowed_file
from src.parser import ResumeParser
from src.preprocess import TextPreprocessor
from src.similarity import SimilarityCalculator
from src.ranking import CandidateRanker

main_bp = Blueprint("main", __name__)

# Initialize components
parser = ResumeParser()
preprocessor = TextPreprocessor()


@main_bp.route("/")
def index():
    """Home page"""
    return render_template("index.html")


@main_bp.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload resumes and job descriptions"""
    if request.method == "POST":
        # Handle resume uploads
        resume_files = request.files.getlist("resumes")
        jd_files = request.files.getlist("job_descriptions")

        uploaded_resumes = []
        uploaded_jds = []

        # Save resumes
        for file in resume_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join("static", "uploads", filename)
                file.save(filepath)
                uploaded_resumes.append(filepath)

        # Save job descriptions
        for file in jd_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join("static", "uploads", filename)
                file.save(filepath)
                uploaded_jds.append(filepath)

        # Process files
        if uploaded_resumes and uploaded_jds:
            # Run screening
            results = run_screening(uploaded_resumes, uploaded_jds)

            # Store results in session or pass to template
            return render_template(
                "results.html",
                results=results,
                resume_count=len(uploaded_resumes),
                jd_count=len(uploaded_jds),
            )

        flash("Please upload valid files", "warning")
        return redirect(url_for("main.upload"))

    return render_template("upload.html")


@main_bp.route("/results")
def results():
    """Show results page"""
    # Load results from file
    results_file = os.path.join("output", "ranked_candidates.csv")
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        return render_template("results.html", results=results_df.to_dict("records"))
    return redirect(url_for("main.upload"))


@main_bp.route("/candidate/<int:candidate_id>")
def candidate_detail(candidate_id):
    """Show candidate details"""
    # Load results
    results_file = os.path.join("output", "ranked_candidates.csv")
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        candidate = results_df.iloc[candidate_id].to_dict()
        return render_template("candidate_detail.html", candidate=candidate)
    return redirect(url_for("main.upload"))


@main_bp.route("/api/rank", methods=["POST"])
def api_rank():
    """API endpoint for ranking"""
    try:
        data = request.json
        resume_text = data.get("resume", "")
        jd_text = data.get("job_description", "")

        if not resume_text or not jd_text:
            return jsonify({"error": "Missing text"}), 400

        # Process texts
        cleaned_resume = preprocessor.get_full_preprocessed_text(resume_text)
        cleaned_jd = preprocessor.get_full_preprocessed_text(jd_text)

        # Calculate similarity
        calculator = SimilarityCalculator()
        similarity = calculator.calculate_similarity_scores(
            [cleaned_resume], [cleaned_jd]
        )

        return jsonify(
            {"similarity_score": float(similarity[0][0] * 100), "status": "success"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@main_bp.route("/download/<filename>")
def download_file(filename):
    """Download results file"""
    filepath = os.path.join("output", filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

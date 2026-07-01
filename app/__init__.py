"""
Flask Application Factory
"""

from flask import Flask
from flask_wtf.csrf import CSRFProtect
import os
import nltk

# Download NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

csrf = CSRFProtect()

# Project root = one level up from this file's folder (app/ -> project root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create_app(config_name="default"):
    """Create Flask application"""
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, "templates"),
        static_folder=os.path.join(BASE_DIR, "static"),
    )

    # Load configuration
    from config import config

    app.config.from_object(config[config_name])

    # Initialize extensions
    csrf.init_app(app)

    # Register blueprints
    from app.routes import main_bp

    app.register_blueprint(main_bp)

    return app

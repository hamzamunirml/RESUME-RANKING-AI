"""
Configuration for Flask Application
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""

    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"

    # Upload settings
    UPLOAD_FOLDER = os.path.join("static", "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {"pdf", "txt"}

    # Data paths
    DATA_DIR = "data"
    RESUME_DIR = os.path.join(DATA_DIR, "resumes")
    JD_DIR = os.path.join(DATA_DIR, "job_descriptions")
    OUTPUT_DIR = "output"

    # Create directories
    for directory in [UPLOAD_FOLDER, RESUME_DIR, JD_DIR, OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    ENV = "development"


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False
    ENV = "production"


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}

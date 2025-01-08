# config/settings.py
import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Model settings
MODEL_PATH = BASE_DIR / "app" / "models" / "ml" / "best_model.keras"
MODEL_INPUT_SHAPE = (224, 224, 3)
CLASS_NAMES = ['Dementia', 'Non Demented', 'Very mild Dementia']

# API settings
API_V1_PREFIX = "/api"
PROJECT_NAME = "Dementia Classification API"
VERSION = "1.0.0"
DESCRIPTION = "API for classifying brain MRI scans for dementia detection"

# CORS settings
ALLOWED_HOSTS = ["*"]  # In production, replace with actual hosts

# Logging settings
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "api.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# File upload settings
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
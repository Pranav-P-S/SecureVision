"""
SecureVision Configuration Module

Centralized configuration for the fire and smoke detection application.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.resolve()

# Model configuration
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "trained_model_l.h5"

# For Streamlit Cloud deployment, check alternate path
CLOUD_MODEL_PATH = Path("/mount/src/securevision/models/trained_model_l.h5")

def get_model_path():
    """Get the appropriate model path based on environment."""
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    elif CLOUD_MODEL_PATH.exists():
        return str(CLOUD_MODEL_PATH)
    else:
        # Fallback to data folder (legacy)
        legacy_path = BASE_DIR / "data" / "trained_model_l.h5"
        if legacy_path.exists():
            return str(legacy_path)
        raise FileNotFoundError(
            f"Model not found. Please ensure the model file exists at: {MODEL_PATH}"
        )

# Detection configuration
DETECTION_CONFIG = {
    "img_size": 224,
    "confidence_threshold": 0.4,
    "labels": {
        0: "Safe",
        1: "🔥 Fire Detected",
        2: "💨 Smoke Detected"
    },
    "label_colors": {
        0: "#22c55e",  # Green for safe
        1: "#ef4444",  # Red for fire
        2: "#f97316"   # Orange for smoke
    }
}

# Assets
ASSETS_DIR = BASE_DIR / "assets"
ALERT_SOUND_PATH = ASSETS_DIR / "alert.wav"

# UI Configuration
UI_CONFIG = {
    "page_title": "SecureVision - Fire & Smoke Detection",
    "page_icon": "🔥",
    "layout": "wide",
    "theme": {
        "primary_color": "#ef4444",
        "background_color": "#0f172a",
        "secondary_background": "#1e293b",
        "text_color": "#f8fafc"
    }
}

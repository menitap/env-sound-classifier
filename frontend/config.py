"""frontend.config

Configuration constants, paths, and settings for the Streamlit application.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configurable Model Directory (Override via MODEL_DIR env var if needed)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "saved_models"))
BASELINE_CNN_PATH = Path(
    os.environ.get("BASELINE_CNN_PATH", MODEL_DIR / "baseline_cnn.keras")
)
CNN_BIGRU_PATH = Path(
    os.environ.get("CNN_BIGRU_PATH", MODEL_DIR / "cnn_bigru.keras")
)

# Configurable Results Directory (Override via RESULTS_DIR env var if needed)
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJECT_ROOT / "results"))
METRICS_PATH = Path(
    os.environ.get("METRICS_PATH", RESULTS_DIR / "model_comparison.json")
)
CONFUSION_MATRIX_PATH = Path(
    os.environ.get("CONFUSION_MATRIX_PATH", RESULTS_DIR / "confusion_matrices.json")
)
NOISE_EVAL_PATH = Path(
    os.environ.get("NOISE_EVAL_PATH", RESULTS_DIR / "noise_robustness.json")
)

# Optional UrbanSound8K dataset audio path for quick sample selection
URBANSOUND8K_AUDIO_DIR = Path(
    os.environ.get("URBANSOUND8K_AUDIO_DIR", r"C:\datasets\UrbanSound8K\audio")
)

# Audio Parameters (Must match preprocessing.py)
TARGET_SAMPLE_RATE = 22050
TARGET_DURATION = 4  # seconds
TARGET_SAMPLES = TARGET_SAMPLE_RATE * TARGET_DURATION  # 88200

# UrbanSound8K Class Mapping (Class ID 0 to 9)
CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]

CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}

# Class Descriptions / Icon helpers for rich display
CLASS_METADATA = {
    "air_conditioner": {"icon": "❄️", "category": "Mechanical"},
    "car_horn": {"icon": "🚗", "category": "Traffic / Urban"},
    "children_playing": {"icon": "👶", "category": "Human / Social"},
    "dog_bark": {"icon": "🐕", "category": "Animal"},
    "drilling": {"icon": "🛠️", "category": "Construction"},
    "engine_idling": {"icon": "🚜", "category": "Traffic / Urban"},
    "gun_shot": {"icon": "💥", "category": "Hazard / Impact"},
    "jackhammer": {"icon": "🔨", "category": "Construction"},
    "siren": {"icon": "🚨", "category": "Emergency"},
    "street_music": {"icon": "🎷", "category": "Music / Social"},
}

# SNR Levels for noise robustness evaluation
SNR_LEVELS = [20, 10, 5]

# Theme Colors
COLORS = {
    "background": "#0E1117",
    "surface": "#1A1D24",
    "surface_accent": "#262B35",
    "primary": "#00D2FF",
    "secondary": "#7928CA",
    "accent": "#00E5FF",
    "text": "#E2E8F0",
    "text_muted": "#94A3B8",
    "border": "#2D3748",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
}


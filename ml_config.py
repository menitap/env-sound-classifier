"""
ml_config.py
Configuration constants for the classical ML extension (SVM and Random Forest).
This module has NO dependency on TensorFlow, Keras, or any CNN-related code.
"""

import hashlib
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

DATASET_ROOT = Path(
    os.environ.get("URBANSOUND8K_PATH", r"C:\Users\madhu\OneDrive\Desktop\datasets")
)
METADATA_PATH = DATASET_ROOT / "UrbanSound8K.csv"
AUDIO_ROOT = DATASET_ROOT

FEATURE_CACHE_DIR = PROJECT_DIR / "feature_cache"
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"
RESULTS_DIR = PROJECT_DIR / "results"

# ---------------------------------------------------------------------------
# Audio / preprocessing constants (must match preprocessing.py exactly)
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 22050
DURATION: int = 4
NUM_SAMPLES: int = SAMPLE_RATE * DURATION  # 88200

# ---------------------------------------------------------------------------
# Class labels (must match config.py CLASS_NAMES order)
# ---------------------------------------------------------------------------
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
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Fold assignments (must match preprocessing.py create_data_splits)
# ---------------------------------------------------------------------------
TRAIN_FOLDS = list(range(1, 9))   # folds 1-8
VAL_FOLDS   = [9]                  # fold 9 → validation
TEST_FOLDS  = [10]                 # fold 10 → final test (never touched during tuning)

# ---------------------------------------------------------------------------
# Feature extraction parameters (250-dimensional vector)
# ---------------------------------------------------------------------------
N_MFCC           = 40    # MFCC coefficients  → 40 mean + 40 delta + 40 delta-delta = 120
N_CHROMA         = 12    # Chroma STFT        → 12
N_CONTRAST_BANDS = 6     # Spectral contrast  → n_bands+1 = 7 rows
CONTRAST_FMIN    = 100   # Hz; 100*2^6=6400 ≤ 11025 Hz Nyquist
N_MEL            = 54    # Mel bands          → 54 mean + 54 var = 108
# ZCR mean (1) + RMS mean (1) + RMS var (1) = 3
# Total: 40+40+40+12+7+54+54+1+1+1 = 250
FEATURE_DIM      = 250

# ---------------------------------------------------------------------------
# Noise SNR levels
# ---------------------------------------------------------------------------
SNR_LEVELS = [20, 10, 5]   # dB

# ---------------------------------------------------------------------------
# Reproducibility seeds
# ---------------------------------------------------------------------------
AUGMENT_SEED = 42
NOISE_SEED   = 99
RF_SEED      = 42
SVM_SEED     = 42

def make_seed(base_seed: int, purpose: str, fold: int,
              posix_path: str, index_or_snr: int) -> int:
    """
    Deterministic, cross-process-stable integer seed derived from hashlib.sha256.
    Never uses Python's built-in hash() which is PYTHONHASHSEED-randomised.
    """
    key = f"{base_seed}|{purpose}|{fold}|{posix_path}|{index_or_snr}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")  # [0, 2**32)

# ---------------------------------------------------------------------------
# Cache filenames
# ---------------------------------------------------------------------------
CACHE_TRAIN_ORIGINAL  = FEATURE_CACHE_DIR / "features_train_original.npz"
CACHE_TRAIN_AUGMENTED = FEATURE_CACHE_DIR / "features_train_augmented.npz"
CACHE_VAL             = FEATURE_CACHE_DIR / "features_val.npz"
CACHE_TEST            = FEATURE_CACHE_DIR / "features_test.npz"

WAVEFORM_CACHE_TEMPLATE = str(FEATURE_CACHE_DIR / "noisy_waveforms_snr{snr}.npy")

NAN_REPORT_PATH = FEATURE_CACHE_DIR / "nan_report.json"

# ---------------------------------------------------------------------------
# Feature configuration hash — stored in every cache for integrity checks
# ---------------------------------------------------------------------------
_FEATURE_CONFIG_STR = (
    f"N_MFCC={N_MFCC}|N_CHROMA={N_CHROMA}|N_CONTRAST_BANDS={N_CONTRAST_BANDS}"
    f"|CONTRAST_FMIN={CONTRAST_FMIN}|N_MEL={N_MEL}|FEATURE_DIM={FEATURE_DIM}"
    f"|SR={SAMPLE_RATE}|DUR={DURATION}"
)
FEATURE_CONFIG_HASH = hashlib.sha256(
    _FEATURE_CONFIG_STR.encode("utf-8")
).hexdigest()[:16]

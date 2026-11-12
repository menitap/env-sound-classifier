"""frontend.model_loader

Handles model weight discovery and loading without mock predictions or fabricated metrics.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from frontend.config import (
    BASELINE_CNN_PATH,
    CNN_BIGRU_PATH,
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    NOISE_EVAL_PATH,
)


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Inspect the filesystem to discover which trained model weights exist.

    Returns
    -------
    dict
        Dictionary containing model metadata and availability status.
    """
    models_info = {
        "Baseline 1D CNN": {
            "path": BASELINE_CNN_PATH,
            "exists": BASELINE_CNN_PATH.is_file(),
            "description": "3-block 1D Convolutional Neural Network with Batch Normalization and Global Average Pooling.",
        },
        "Hybrid CNN-BiGRU": {
            "path": CNN_BIGRU_PATH,
            "exists": CNN_BIGRU_PATH.is_file(),
            "description": "3-block 1D CNN feature extractor coupled with Bidirectional GRU for temporal dynamics.",
        },
    }
    return models_info


def load_model_if_available(model_name: str) -> Optional[Any]:
    """Load a trained Keras model if its weight file exists on disk.

    Returns None if weights are not present. Never generates mock models.

    Parameters
    ----------
    model_name : str
        'Baseline 1D CNN' or 'Hybrid CNN-BiGRU'

    Returns
    -------
    model : Optional[tf.keras.Model]
        Loaded model object or None.
    """
    import os

    models_info = get_available_models()
    if model_name not in models_info:
        return None

    path: Path = models_info[model_name]["path"]
    if not path.is_file():
        return None

    try:
        import tensorflow as tf

        return tf.keras.models.load_model(str(path))
    except Exception as exc:
        print(f"Error loading model from {path}: {exc}")
        return None


def run_inference(
    model: Any, input_tensor: np.ndarray
) -> Optional[Tuple[int, float, np.ndarray]]:
    """Run model inference on preprocessed audio input.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded Keras model.
    input_tensor : np.ndarray
        Array of shape (1, 88200, 1) and dtype float32.

    Returns
    -------
    pred_class_id : int
        Predicted integer class ID (0 to 9).
    confidence : float
        Softmax probability of the top class (0.0 to 1.0).
    probabilities : np.ndarray
        1D array of shape (10,) with softmax class probabilities.
    """
    if model is None:
        return None

    probs = model.predict(input_tensor, verbose=0)[0]
    pred_class_id = int(np.argmax(probs))
    confidence = float(probs[pred_class_id])

    return pred_class_id, confidence, probs


def load_evaluation_metrics() -> Optional[Dict[str, Any]]:
    """Load actual model evaluation metrics from results directory.

    Returns None if metrics have not been generated yet. Never fabricates values.
    """
    import json

    if not METRICS_PATH.is_file():
        return None

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_confusion_matrices() -> Optional[Dict[str, Any]]:
    """Load actual confusion matrix data from results directory.

    Returns None if not generated yet.
    """
    import json

    if not CONFUSION_MATRIX_PATH.is_file():
        return None

    try:
        with open(CONFUSION_MATRIX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_noise_evaluation() -> Optional[Dict[str, Any]]:
    """Load actual noise robustness evaluation results.

    Returns None if not generated yet.
    """
    import json

    if not NOISE_EVAL_PATH.is_file():
        return None

    try:
        with open(NOISE_EVAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


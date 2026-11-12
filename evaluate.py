"""evaluate.py

Evaluation pipeline for the Environmental Sound Classification project.

Loads each trained model from saved_models/ and evaluates it on the
held-out test set (fold 10 of UrbanSound8K).

Outputs (all written to results/):
  model_comparison.json         Overall metrics for both models (keys: accuracy, precision, recall, f1_score)
  confusion_matrices.json       Confusion matrix 2D list for each model
  noise_robustness.json         Accuracy list across SNR levels [Clean, 20, 10, 5]
  <model>_classification_report.txt  Per-class precision/recall/F1
  confusion_matrix_<model>.png  Heatmap figure
  accuracy_loss_curves_<model>.png  Training history curves

Usage:
    python evaluate.py                   # evaluate both models
    python evaluate.py --model baseline
    python evaluate.py --model bigru
    python evaluate.py --no-noise        # skip noise robustness test
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Fix Matplotlib cache warning on Windows
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# ---------------------------------------------------------------------------
# Paths (mirror train.py exactly)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

DATASET_ROOT = Path(
    os.environ.get("URBANSOUND8K_PATH", r"C:\datasets\UrbanSound8K")
)
METADATA_PATH = DATASET_ROOT / "metadata" / "UrbanSound8K.csv"
AUDIO_ROOT = DATASET_ROOT / "audio"

SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"
RESULTS_DIR = PROJECT_DIR / "results"

BASELINE_MODEL_PATH = SAVED_MODELS_DIR / "baseline_cnn.keras"
BIGRU_MODEL_PATH = SAVED_MODELS_DIR / "cnn_bigru.keras"

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music",
]

INPUT_SHAPE = (88200, 1)
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        print(
            "\n[ERROR] TensorFlow is not installed.\n"
            "Install with:  pip install tensorflow\n"
        )
        sys.exit(1)


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[WARN] matplotlib not installed – plots will be skipped.")
        return None


def _require_sklearn():
    try:
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            precision_recall_fscore_support, accuracy_score,
        )
        return classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
    except ImportError:
        print("[ERROR] scikit-learn is required for evaluation.")
        sys.exit(1)


def _load_test_data(batch_size=32):
    """Load and preprocess all test-fold samples into NumPy arrays.

    Returns X_test (N, 88200, 1) float32 and y_test (N,) int64.
    """
    import numpy as np
    from preprocessing import load_metadata, create_data_splits, get_audio_path, load_and_preprocess_audio

    if not METADATA_PATH.is_file():
        print(f"[ERROR] Metadata not found: {METADATA_PATH}")
        sys.exit(1)
    if not AUDIO_ROOT.is_dir():
        print(f"[ERROR] Audio root not found: {AUDIO_ROOT}")
        sys.exit(1)

    metadata = load_metadata(METADATA_PATH)
    _, _, test_df = create_data_splits(metadata)

    print(f"  Test samples (fold 10): {len(test_df)}")
    X_list, y_list = [], []

    for i, (_, row) in enumerate(test_df.iterrows()):
        audio_path = get_audio_path(AUDIO_ROOT, row["fold"], row["slice_file_name"])
        try:
            processed = load_and_preprocess_audio(audio_path)
            X_list.append(processed)
            y_list.append(int(row["classID"]))
        except Exception as exc:
            print(f"  [WARN] Skipping {row['slice_file_name']}: {exc}")

        if (i + 1) % 200 == 0:
            print(f"  Loaded {i + 1}/{len(test_df)} test files...")

    X_test = np.array(X_list, dtype=np.float32)
    y_test = np.array(y_list, dtype=np.int64)
    print(f"  Test array shape: {X_test.shape}  labels: {y_test.shape}")
    return X_test, y_test


def _load_training_meta(model_name):
    """Load training metadata saved by train.py, if available."""
    meta_path = RESULTS_DIR / f"{model_name}_training_meta.json"
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_history(model_name):
    """Load training history saved by train.py, if available."""
    hist_path = RESULTS_DIR / f"{model_name}_history.json"
    if hist_path.is_file():
        with open(hist_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm, model_display_name, out_path):
    """Save a heatmap of the confusion matrix to out_path."""
    plt = _require_matplotlib()
    if plt is None:
        return

    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(NUM_CLASSES),
        yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title=f"Confusion Matrix — {model_display_name}",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix plot saved → {out_path}")


def _plot_training_curves(history, model_display_name, out_path):
    """Save accuracy + loss curves from training history."""
    plt = _require_matplotlib()
    if plt is None or history is None:
        return

    epochs_range = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, history.get("accuracy", []), label="Train Accuracy")
    if "val_accuracy" in history:
        ax1.plot(epochs_range, history["val_accuracy"], label="Val Accuracy")
    ax1.set(title=f"Accuracy — {model_display_name}", xlabel="Epoch", ylabel="Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["loss"], label="Train Loss")
    if "val_loss" in history:
        ax2.plot(epochs_range, history["val_loss"], label="Val Loss")
    ax2.set(title=f"Loss — {model_display_name}", xlabel="Epoch", ylabel="Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  Training curves saved → {out_path}")


# ---------------------------------------------------------------------------
# Noise robustness
# ---------------------------------------------------------------------------

def _evaluate_noise_robustness(model, X_test, y_test):
    """Evaluate model at 20, 10, 5 dB SNR using noisy_testing.py logic.

    Returns dict mapping snr integer -> accuracy float.
    """
    from noisy_testing import evaluate_noise_levels

    print("\n  Running noise robustness evaluation (20, 10, 5 dB)...")
    try:
        results = evaluate_noise_levels(model, X_test, y_test)
        return results
    except Exception as exc:
        print(f"  [WARN] Noise evaluation failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_key, model_display_name, model_path, X_test, y_test, run_noise):
    """Evaluate one model on test data and return metrics."""
    import numpy as np

    tf = _require_tensorflow()
    classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score = _require_sklearn()

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_display_name}")
    print(f"{'='*60}")

    if not model_path.is_file():
        short_key = model_key.split("_")[0]
        print(
            f"\n  [SKIP] Model weights not found at: {model_path}\n"
            f"  Run 'python train.py --model {short_key}' first.\n"
        )
        return None

    print(f"  Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as exc:
        print(f"  [ERROR] Could not load model: {exc}")
        return None

    model.summary(line_length=80)

    # Predict
    print("\n  Running inference on fold 10 test set...")
    t0 = time.time()
    y_pred_proba = model.predict(X_test, batch_size=32, verbose=1)
    inference_time = time.time() - t0
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Metrics
    test_acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    test_loss, test_acc_keras = model.evaluate(X_test, y_test, batch_size=32, verbose=0)

    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Precision (w) : {precision:.4f}")
    print(f"  Recall (w)    : {recall:.4f}")
    print(f"  F1-score (w)  : {f1:.4f}")
    print(f"  Inference time: {inference_time:.2f} s")
    print(f"\n{report}")

    training_meta = _load_training_meta(model_key)
    history = _load_history(model_key)

    # Save classification report text
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"{model_key}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_display_name}\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n\n")
        f.write(report)
    print(f"  Classification report saved → {report_path}")

    # Plot confusion matrix
    cm_plot_path = RESULTS_DIR / f"confusion_matrix_{model_key}.png"
    _plot_confusion_matrix(cm, model_display_name, cm_plot_path)

    # Plot training curves
    if history:
        curves_path = RESULTS_DIR / f"accuracy_loss_curves_{model_key}.png"
        _plot_training_curves(history, model_display_name, curves_path)

    # Noise robustness
    noise_results = {}
    if run_noise:
        noise_results = _evaluate_noise_robustness(model, X_test, y_test)

    # Return structure matching frontend expectations exactly
    result = {
        "model_name": model_display_name,
        "model_key": model_key,
        "model_path": str(model_path),
        "loss": round(float(test_loss), 4),
        "accuracy": round(float(test_acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "inference_time_sec": round(inference_time, 3),
        "noise_results": noise_results,
        "training_meta": training_meta,
    }

    return result, cm.tolist()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_all_results(eval_results):
    """Write model_comparison.json, confusion_matrices.json, noise_robustness.json.

    Keys and formats strictly adhere to frontend/views/comparison.py & noise_robustness.py:
      - model_comparison.json:
          {"baseline_cnn": {"accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...},
           "cnn_bigru":    {"accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...}}
      - confusion_matrices.json:
          {"baseline_cnn": [[...], ...], "cnn_bigru": [[...], ...]}
      - noise_robustness.json:
          {"snr_levels": ["Clean", 20, 10, 5],
           "baseline_cnn": [clean_acc, snr20_acc, snr10_acc, snr5_acc],
           "cnn_bigru":    [clean_acc, snr20_acc, snr10_acc, snr5_acc]}
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. model_comparison.json
    comp_data = {}
    for key, (result, _) in eval_results.items():
        comp_data[key] = {
            "model_name": result["model_name"],
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1_score": result["f1_score"],
            "loss": result["loss"],
            "inference_time_sec": result["inference_time_sec"],
        }
    comp_path = RESULTS_DIR / "model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)
    print(f"\n  Model comparison saved → {comp_path}")

    # 2. confusion_matrices.json
    cm_dict = {key: cm for key, (_, cm) in eval_results.items()}
    cm_path = RESULTS_DIR / "confusion_matrices.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2)
    print(f"  Confusion matrices saved → {cm_path}")

    # 3. noise_robustness.json
    noise_dict = {
        "snr_levels": ["Clean", 20, 10, 5]
    }
    for key, (result, _) in eval_results.items():
        clean_acc = result["accuracy"]
        nr = result.get("noise_results")
        if nr:
            # Map values in exact order: Clean, 20 dB, 10 dB, 5 dB
            # Keep null/None if a specific SNR level is missing rather than fabricating 0.0
            acc_list = [
                clean_acc,
                round(float(nr[20]), 4) if 20 in nr else (round(float(nr["20"]), 4) if "20" in nr else None),
                round(float(nr[10]), 4) if 10 in nr else (round(float(nr["10"]), 4) if "10" in nr else None),
                round(float(nr[5]), 4) if 5 in nr else (round(float(nr["5"]), 4) if "5" in nr else None),
            ]
        else:
            acc_list = [clean_acc, None, None, None]
        noise_dict[key] = acc_list

    noise_path = RESULTS_DIR / "noise_robustness.json"
    with open(noise_path, "w", encoding="utf-8") as f:
        json.dump(noise_dict, f, indent=2)
    print(f"  Noise robustness saved → {noise_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on UrbanSound8K test fold 10."
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "bigru", "both"],
        default="both",
        help="Which model(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Skip noise robustness evaluation.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tf = _require_tensorflow()
    print(f"TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus if gpus else 'None (CPU only)'}")

    run_noise = not args.no_noise

    MODELS = {
        "baseline_cnn": ("Baseline 1D CNN", BASELINE_MODEL_PATH),
        "cnn_bigru":    ("Hybrid CNN-BiGRU", BIGRU_MODEL_PATH),
    }

    if args.model == "baseline":
        MODELS = {"baseline_cnn": MODELS["baseline_cnn"]}
    elif args.model == "bigru":
        MODELS = {"cnn_bigru": MODELS["cnn_bigru"]}

    # Check if requested model exists
    existing_models = {k: v for k, v in MODELS.items() if v[1].is_file()}
    if not existing_models:
        print("\n[INFO] No trained model weights found in saved_models/ for evaluation.")
        for k, (name, path) in MODELS.items():
            print(f"  {name}: {path} (missing)")
        print("\nRun 'python train.py' to train models first.\n")
        sys.exit(0)

    # Load test data once
    print("\nLoading test data (fold 10)...")
    X_test, y_test = _load_test_data(batch_size=args.batch)

    if len(X_test) == 0:
        print("[ERROR] No test samples could be loaded.")
        sys.exit(1)

    eval_results = {}
    for model_key, (display_name, model_path) in existing_models.items():
        out = evaluate_model(
            model_key=model_key,
            model_display_name=display_name,
            model_path=model_path,
            X_test=X_test,
            y_test=y_test,
            run_noise=run_noise,
        )
        if out is not None:
            eval_results[model_key] = out

    if eval_results:
        _save_all_results(eval_results)

    # Summary
    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")
    for key, (result, _) in eval_results.items():
        print(f"\n  {result['model_name']}")
        print(f"    Test Accuracy : {result['accuracy']:.4f}")
        print(f"    Test Loss     : {result['loss']:.4f}")
        print(f"    F1 (weighted) : {result['f1_score']:.4f}")
        if result["noise_results"]:
            for snr, acc in result["noise_results"].items():
                print(f"    Noisy @ {snr:>2} dB : {acc:.4f}")
    print(f"\n  Results written to: {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()

"""
ml_evaluate.py
Phase F/G: Evaluate all models on clean fold-10 test set and noisy waveforms.

Runs in .venv_ml. Zero dependency on TensorFlow/Keras.
Fold 10 is loaded HERE and ONLY HERE — after all tuning is frozen.

Outputs per model:
  results/<model>_test_metrics.json        — clean fold-10 test metrics
  results/<model>_noise_robustness.json    — per-SNR metrics
  results/<model>_classification_report.txt
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from ml_config import (
    FEATURE_CACHE_DIR, SAVED_MODELS_DIR, RESULTS_DIR,
    CLASS_NAMES, NUM_CLASSES, SNR_LEVELS, FEATURE_CONFIG_HASH,
    WAVEFORM_CACHE_TEMPLATE, make_seed, NOISE_SEED,
)
from feature_extraction import extract_features

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cache loader with integrity check
# ---------------------------------------------------------------------------
def load_cache(path: Path) -> dict:
    d = dict(np.load(path, allow_pickle=True))
    stored_hash = str(d["feature_config_hash"].item())
    if stored_hash != FEATURE_CONFIG_HASH:
        raise ValueError(
            f"Feature config hash mismatch in {path}\n"
            f"  Expected: {FEATURE_CONFIG_HASH}\n  Got: {stored_hash}"
        )
    return d


# ---------------------------------------------------------------------------
# Anomaly gate: test accuracy >95% → likely data leakage (D23)
# ---------------------------------------------------------------------------
def _leakage_gate(acc: float, model_name: str):
    if acc > 0.95:
        raise RuntimeError(
            f"[LEAKAGE GATE TRIGGERED] {model_name} test_accuracy={acc:.4f} > 0.95. "
            "Halt immediately and inspect for data leakage."
        )


# ---------------------------------------------------------------------------
# Load sklearn model (SVM pipeline or RF)
# ---------------------------------------------------------------------------
def _load_sklearn(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Evaluate an sklearn model on clean test features
# ---------------------------------------------------------------------------
def evaluate_sklearn_clean(model_name: str, model, X_test, y_test):
    """Uses model.predict() — never np.argmax (D17)."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    _leakage_gate(acc, model_name)

    report_str = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model": model_name,
        "split": "test",
        "fold": 10,
        "test_accuracy": float(acc),
        "confusion_matrix": cm,
    }

    out_metrics = RESULTS_DIR / f"{model_name}_test_metrics.json"
    out_report  = RESULTS_DIR / f"{model_name}_classification_report.txt"

    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_report, "w") as f:
        f.write(report_str)

    print(f"  {model_name}  test_accuracy={acc:.4f}")
    print(f"  Saved: {out_metrics}")
    print(f"  Saved: {out_report}")
    return acc


# ---------------------------------------------------------------------------
# Evaluate an sklearn model on noisy waveforms (extract features on the fly)
# ---------------------------------------------------------------------------
def evaluate_sklearn_noisy(model_name: str, model, fn_te, fo_te, y_test,
                            train_mean: np.ndarray):
    """
    Loads noisy waveform caches, extracts features, runs model.predict().
    Uses the training-set mean for any non-finite values (same imputer as Phase E).
    """
    noise_results = {}

    for snr in SNR_LEVELS:
        wav_path = Path(WAVEFORM_CACHE_TEMPLATE.format(snr=snr))
        if not wav_path.exists():
            print(f"  [WARN] Noisy waveform cache missing: {wav_path}")
            continue

        X_noisy_wav = np.load(wav_path)  # (N, 88200)
        N = len(X_noisy_wav)
        X_feat = np.zeros((N, 250), dtype=np.float32)

        for i in range(N):
            f = extract_features(X_noisy_wav[i])
            bad = ~np.isfinite(f)
            if bad.any():
                f[bad] = train_mean[bad]
            X_feat[i] = f

        y_pred = model.predict(X_feat)
        acc = accuracy_score(y_test, y_pred)
        noise_results[f"snr_{snr}db"] = {
            "snr_db": snr,
            "test_accuracy": float(acc),
        }
        print(f"    SNR={snr:2d} dB → accuracy={acc:.4f}")

    out_path = RESULTS_DIR / f"{model_name}_noise_robustness.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "split": "test",
            "fold": 10,
            "results": noise_results,
        }, f, indent=2)
    print(f"  Saved: {out_path}")
    return noise_results


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------
def evaluate_final_models():
    """
    Load test cache (fold 10) HERE and only here — after all tuning is frozen.
    Runs clean + noisy evaluation for all 4 sklearn models.
    """
    print("\n" + "=" * 60)
    print("PHASE F/G: FINAL MODEL EVALUATION")
    print("=" * 60)
    print("Loading fold-10 test features (held-out; never used in tuning)...")

    test_data = load_cache(FEATURE_CACHE_DIR / "features_test.npz")
    X_test = test_data["features"].astype(np.float64)
    y_test = test_data["labels"].astype(np.int32)
    fn_te  = test_data["filenames"]
    fo_te  = test_data["folds"]

    print(f"Test set: {X_test.shape[0]} clips, {X_test.shape[1]} features")

    # Training-set mean for noisy imputation (computed from clean original features)
    train_data = load_cache(FEATURE_CACHE_DIR / "features_train_original.npz")
    train_mean = np.nanmean(train_data["features"].astype(np.float32), axis=0)

    summary_rows = []

    models_to_eval = [
        ("svm_original",   SAVED_MODELS_DIR / "svm_original.joblib",   np.float64),
        ("svm_augmented",  SAVED_MODELS_DIR / "svm_augmented.joblib",  np.float64),
        ("rf_original",    SAVED_MODELS_DIR / "rf_original.joblib",    np.float32),
        ("rf_augmented",   SAVED_MODELS_DIR / "rf_augmented.joblib",   np.float32),
    ]

    for model_name, model_path, dtype in models_to_eval:
        print(f"\n--- {model_name} ---")
        try:
            model = _load_sklearn(model_path)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        X_eval = X_test.astype(dtype)

        # D15 runtime confirmation: test data loaded strictly here, after all tuning
        clean_acc = evaluate_sklearn_clean(model_name, model, X_eval, y_test)
        noise_res = evaluate_sklearn_noisy(model_name, model, fn_te, fo_te,
                                           y_test, train_mean)

        summary_rows.append({
            "model": model_name,
            "test_accuracy_clean": float(clean_acc),
            **{f"test_acc_snr{snr}db": noise_res.get(f"snr_{snr}db", {}).get("test_accuracy")
               for snr in SNR_LEVELS}
        })

    # Save experiment comparison summary (new file — does not touch results/noise_robustness.json)
    summary_path = RESULTS_DIR / "experiment_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSaved: {summary_path}")
    print("\n[EVALUATION COMPLETE]")


if __name__ == "__main__":
    evaluate_final_models()

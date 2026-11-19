"""
evaluate_cnn_augmented.py
Phase G: Evaluate the augmented CNN on clean fold-10 and noisy waveforms.

Runs in .venv (TensorFlow environment).
Uses np.argmax on Keras softmax outputs (D17: argmax only on Keras outputs).
Never modifies results/noise_robustness.json — writes to new files.
"""

import json
import pathlib
import sys

import numpy as np

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from preprocessing import (
    load_metadata, create_data_splits, get_audio_path, load_and_preprocess_audio
)
from augmentation import add_noise
from ml_config import (
    DATASET_ROOT, METADATA_PATH, AUDIO_ROOT,
    RESULTS_DIR, SAVED_MODELS_DIR, FEATURE_CACHE_DIR,
    SNR_LEVELS, CLASS_NAMES, WAVEFORM_CACHE_TEMPLATE, NOISE_SEED, make_seed
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _leakage_gate(acc, model_name):
    if acc > 0.95:
        raise RuntimeError(
            f"[LEAKAGE GATE] {model_name} test_accuracy={acc:.4f} > 0.95. "
            "Halt and inspect for data leakage."
        )


def evaluate_cnn_augmented():
    import tensorflow as tf

    model_path = SAVED_MODELS_DIR / "baseline_cnn_augmented.keras"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Run train_cnn_augmented.py first.")
        sys.exit(1)

    print("=" * 60)
    print("PHASE G: EVALUATE AUGMENTED CNN")
    print("=" * 60)

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded: {model_path}")

    # -- Load test waveforms (fold 10) --
    clean_wav_path = FEATURE_CACHE_DIR / "clean_waveforms_test.npy"
    clean_meta_path = FEATURE_CACHE_DIR / "clean_meta_test.npz"
    if clean_wav_path.exists() and clean_meta_path.exists():
        print(f"Loading cached clean test waveforms: {clean_wav_path}")
        X_test = np.load(clean_wav_path)[:, :, np.newaxis]
        y_test = np.load(clean_meta_path)["labels"]
    else:
        meta = load_metadata(METADATA_PATH)
        _, _, test_df = create_data_splits(meta)

        X_test, y_test = [], []
        for _, row in test_df.iterrows():
            p = get_audio_path(AUDIO_ROOT, row["fold"], row["slice_file_name"])
            try:
                wav = load_and_preprocess_audio(p)
                X_test.append(wav)
                y_test.append(int(row["classID"]))
            except Exception as exc:
                print(f"  [WARN] skip {row['slice_file_name']}: {exc}")

        X_test = np.array(X_test, dtype=np.float32)   # (N, 88200, 1)
        y_test = np.array(y_test, dtype=np.int32)
    print(f"Test set: {len(X_test)} clips")

    # -- Clean evaluation --
    y_pred_proba = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)        # D17: argmax on Keras output
    acc = accuracy_score(y_test, y_pred)
    _leakage_gate(acc, "baseline_cnn_augmented")

    report_str = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model": "baseline_cnn_augmented",
        "split": "test",
        "fold": 10,
        "test_accuracy": float(acc),
        "confusion_matrix": cm,
    }

    out_metrics = RESULTS_DIR / "baseline_cnn_augmented_test_metrics.json"
    out_report  = RESULTS_DIR / "baseline_cnn_augmented_classification_report.txt"

    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_report, "w") as f:
        f.write(report_str)

    print(f"\ntest_accuracy (clean) = {acc:.4f}")
    print(f"Saved: {out_metrics}")
    print(f"Saved: {out_report}")

    # -- Noisy evaluation (use cached noisy waveforms if available) --
    noise_results = {}
    for snr in SNR_LEVELS:
        wav_path = pathlib.Path(WAVEFORM_CACHE_TEMPLATE.format(snr=snr))
        if wav_path.exists():
            X_noisy = np.load(wav_path)             # (N, 88200)
            X_noisy = X_noisy[:, :, np.newaxis]     # (N, 88200, 1)
        else:
            # Generate on the fly if cache missing (should not happen after Phase E)
            print(f"  [WARN] Noisy cache missing for SNR={snr}, generating on the fly...")
            meta2 = load_metadata(METADATA_PATH)
            _, _, test_df2 = create_data_splits(meta2)
            X_noisy_list = []
            for _, row in test_df2.iterrows():
                p = get_audio_path(AUDIO_ROOT, row["fold"], row["slice_file_name"])
                try:
                    wav = load_and_preprocess_audio(p).squeeze()
                    seed = make_seed(NOISE_SEED, "noise", int(row["fold"]),
                                     row["slice_file_name"], snr)
                    prev = np.random.get_state()
                    try:
                        np.random.seed(seed)
                        noisy = add_noise(wav, snr_db=snr)
                    finally:
                        np.random.set_state(prev)
                    X_noisy_list.append(noisy)
                except Exception:
                    X_noisy_list.append(np.zeros(88200, dtype=np.float32))
            X_noisy = np.array(X_noisy_list, dtype=np.float32)[:, :, np.newaxis]

        y_noisy_proba = model.predict(X_noisy, batch_size=32, verbose=0)
        y_noisy_pred  = np.argmax(y_noisy_proba, axis=1)   # D17
        noisy_acc = accuracy_score(y_test, y_noisy_pred)
        noise_results[f"snr_{snr}db"] = {
            "snr_db": snr,
            "test_accuracy": float(noisy_acc),
        }
        print(f"  SNR={snr:2d} dB → test_accuracy={noisy_acc:.4f}")

    out_noise = RESULTS_DIR / "baseline_cnn_augmented_noise_robustness.json"
    with open(out_noise, "w") as f:
        json.dump({
            "model": "baseline_cnn_augmented",
            "split": "test",
            "fold": 10,
            "results": noise_results,
        }, f, indent=2)
    print(f"Saved: {out_noise}")

    print("\n[PHASE G EVALUATION COMPLETE]")
    return acc


if __name__ == "__main__":
    evaluate_cnn_augmented()

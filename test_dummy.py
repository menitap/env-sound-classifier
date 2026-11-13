"""test_dummy.py

Lightweight dummy-data test suite for the Environmental Sound Classification pipeline.

Tests ONLY use in-memory NumPy arrays — the real UrbanSound8K dataset is
NOT required, and NO actual training of the full model occurs.

What is tested:
  1.  TensorFlow import and version
  2.  Model construction (Baseline CNN + CNN-BiGRU)
  3.  Input shape compatibility  (batch, 88200, 1)
  4.  Forward pass — output shape must be (batch, 10)
  5.  Softmax probabilities sum ≈ 1.0
  6.  Label dtype compatibility (int64 sparse labels)
  7.  Batch generator with an in-memory fake metadata DataFrame
  8.  One mini training step (2 epochs, batch=4, 8 dummy samples)
  9.  Save model to a temp file and reload it
  10. Evaluate-style metrics on dummy predictions (sklearn)
  11. Noise robustness helper (add_noise from augmentation.py)

Usage:
    python test_dummy.py
"""

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
SKIP = f"{YELLOW}SKIP{RESET}"

results = []


def run_test(name, fn):
    print(f"  [{name}]  ", end="", flush=True)
    try:
        fn()
        print(PASS)
        results.append((name, "PASS", None))
    except Exception as exc:
        print(FAIL)
        tb = traceback.format_exc()
        print(f"    {RED}{exc}{RESET}")
        results.append((name, "FAIL", tb))


# ---------------------------------------------------------------------------
# Shared state across tests
# ---------------------------------------------------------------------------
_tf = None
_baseline_model = None
_bigru_model = None
_dummy_X = None   # shape (8, 88200, 1) float32
_dummy_y = None   # shape (8,) int64


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tensorflow_import():
    global _tf
    import tensorflow as tf  # noqa
    _tf = tf
    assert tf.__version__, "TF version string is empty"
    print(f"(TF {tf.__version__}) ", end="")


def test_model_construction():
    global _baseline_model, _bigru_model
    assert _tf is not None, "TensorFlow not imported yet"
    from models import build_baseline_cnn, build_cnn_bigru
    _baseline_model = build_baseline_cnn()
    _bigru_model = build_cnn_bigru()
    assert _baseline_model is not None
    assert _bigru_model is not None


def test_forward_pass_baseline():
    global _dummy_X, _dummy_y
    assert _baseline_model is not None
    rng = np.random.default_rng(0)
    _dummy_X = rng.standard_normal((8, 88200, 1)).astype(np.float32)
    _dummy_y = rng.integers(0, 10, size=(8,), dtype=np.int64)

    out = _baseline_model.predict(_dummy_X[:2], verbose=0)
    assert out.shape == (2, 10), f"Expected (2,10) got {out.shape}"


def test_forward_pass_bigru():
    assert _bigru_model is not None and _dummy_X is not None
    out = _bigru_model.predict(_dummy_X[:2], verbose=0)
    assert out.shape == (2, 10), f"Expected (2,10) got {out.shape}"


def test_softmax_sums_to_one():
    for name, model in [("baseline", _baseline_model), ("bigru", _bigru_model)]:
        out = model.predict(_dummy_X[:2], verbose=0)
        row_sums = out.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            f"{name}: row sums not ≈1: {row_sums}"
        )


def test_label_compatibility():
    # Sparse labels passed to compiled model — should not raise
    assert _baseline_model is not None and _dummy_X is not None
    loss_val, _ = _baseline_model.evaluate(_dummy_X, _dummy_y, batch_size=4, verbose=0)
    assert loss_val > 0, "Loss should be > 0 for random predictions"


def test_batch_generator():
    """Create a tiny fake metadata DataFrame and exercise create_batch_generator."""
    import tempfile, os, soundfile as sf
    from preprocessing import create_batch_generator

    # Build 4 small WAV files in a temp fold directory
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp_root:
        fold_dir = Path(tmp_root) / "fold1"
        fold_dir.mkdir()
        rows = []
        for i in range(4):
            fname = f"dummy_{i}.wav"
            wav_data = rng.standard_normal(22050).astype(np.float32)   # 1 s audio
            sf.write(str(fold_dir / fname), wav_data, 22050)
            rows.append({
                "slice_file_name": fname,
                "fold": 1,
                "classID": i % 10,
                "class": f"class_{i % 10}",
            })

        df = pd.DataFrame(rows)
        batches = list(create_batch_generator(df, tmp_root, batch_size=2, shuffle=False))
        assert len(batches) == 2, f"Expected 2 batches, got {len(batches)}"
        bx, by = batches[0]
        assert bx.shape == (2, 88200, 1), f"Bad X shape: {bx.shape}"
        assert by.shape == (2,), f"Bad y shape: {by.shape}"
        assert bx.dtype == np.float32
        assert by.dtype == np.int64


def test_mini_training_step():
    """Train baseline model for 2 epochs on 8 dummy samples."""
    assert _baseline_model is not None and _dummy_X is not None
    history = _baseline_model.fit(
        _dummy_X, _dummy_y,
        epochs=2,
        batch_size=4,
        verbose=0,
    )
    assert "loss" in history.history
    assert len(history.history["loss"]) == 2


def test_save_and_reload():
    """Save baseline model to a temp path and reload it."""
    assert _baseline_model is not None and _dummy_X is not None
    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "test_baseline.keras"
        _baseline_model.save(str(save_path))
        assert save_path.is_file(), "Saved model file not found"

        reloaded = _tf.keras.models.load_model(str(save_path))
        out_orig = _baseline_model.predict(_dummy_X[:1], verbose=0)
        out_new  = reloaded.predict(_dummy_X[:1], verbose=0)
        assert np.allclose(out_orig, out_new, atol=1e-5), "Reloaded model outputs differ"


def test_sklearn_metrics():
    from sklearn.metrics import (
        accuracy_score, f1_score,
        precision_score, recall_score, confusion_matrix,
    )
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 10, 50, dtype=np.int64)
    y_pred = rng.integers(0, 10, 50, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    pr  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rc  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    assert 0 <= acc <= 1
    assert cm.shape == (10, 10)
    assert f1 >= 0 and pr >= 0 and rc >= 0


def test_noise_augmentation():
    from augmentation import add_noise
    rng = np.random.default_rng(3)
    audio = rng.standard_normal(88200).astype(np.float32)
    for snr in [20, 10, 5]:
        noisy = add_noise(audio, snr_db=snr)
        assert noisy.shape == audio.shape, f"Shape mismatch at {snr} dB"
        assert noisy.dtype == np.float32
        # noisy must differ from clean
        assert not np.allclose(noisy, audio), f"add_noise had no effect at {snr} dB"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("1. TensorFlow import",         test_tensorflow_import),
    ("2. Model construction",         test_model_construction),
    ("3. Forward pass — Baseline",    test_forward_pass_baseline),
    ("4. Forward pass — CNN-BiGRU",   test_forward_pass_bigru),
    ("5. Softmax sums ≈ 1",          test_softmax_sums_to_one),
    ("6. Label compatibility",        test_label_compatibility),
    ("7. Batch generator",            test_batch_generator),
    ("8. Mini training step",         test_mini_training_step),
    ("9. Save & reload model",        test_save_and_reload),
    ("10. Sklearn metrics",           test_sklearn_metrics),
    ("11. Noise augmentation",        test_noise_augmentation),
]


def main():
    print(f"\n{BOLD}=== Dummy-data test suite ==={RESET}")
    print(f"Python {sys.version.split()[0]}\n")

    for name, fn in TESTS:
        run_test(name, fn)

    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")

    print(f"\n{BOLD}--- Results ---{RESET}")
    for name, status, tb in results:
        color = GREEN if status == "PASS" else RED
        print(f"  {color}{status}{RESET}  {name}")
        if tb:
            for line in tb.strip().split("\n")[-4:]:
                print(f"         {line}")

    print(f"\n  {passed}/{len(TESTS)} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
        sys.exit(1)
    else:
        print(f"  — all tests passed ✓")


if __name__ == "__main__":
    main()


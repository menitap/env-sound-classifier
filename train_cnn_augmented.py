"""
train_cnn_augmented.py
Phase G: Train the augmented 1D CNN (Experiment A).

Runs in .venv (TensorFlow environment).
Imports build_baseline_cnn and train_model from existing protected files.
Does NOT modify models.py, train.py, or any existing file.
"""

import os
import pathlib
import sys

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

# Ensure URBANSOUND8K_PATH is set before train.py is imported
os.environ.setdefault("URBANSOUND8K_PATH", r"C:\Users\madhu\OneDrive\Desktop\datasets")

# Import ONLY from existing protected modules — do not copy or alter them.
from models import build_baseline_cnn
from train import train_model

# ---------------------------------------------------------------------------
# Phase G: Train augmented CNN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE G: AUGMENTED CNN TRAINING")
    print("=" * 60)
    print("Architecture : build_baseline_cnn (models.py:9) — unmodified")
    print("Augmentation : augment=True (replacement, N samples, per epoch)")
    print("Validation   : fold 9 (clean, augment=False enforced in train.py)")
    print("Test fold    : fold 10 — discarded during training")
    print()

    history, elapsed = train_model(
        model_name="baseline_cnn_augmented",
        model_builder=build_baseline_cnn,
        model_path=pathlib.Path("saved_models/baseline_cnn_augmented.keras"),
        epochs=50,
        batch_size=32,
        augment=True,
        subset_size=None,
    )

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    best_epoch   = history.history.get("val_accuracy", [0.0]).index(best_val_acc) + 1

    print()
    print("=" * 60)
    print(f"TRAINING COMPLETE — elapsed: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"Best val_accuracy : {best_val_acc:.4f} at epoch {best_epoch}")
    print("Model saved to    : saved_models/baseline_cnn_augmented.keras")
    print("=" * 60)

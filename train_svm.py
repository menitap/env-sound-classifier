"""
train_svm.py
Phase F: Train SVM classifiers (original and augmented features).

Runs in .venv_ml. Zero dependency on TensorFlow.
Fold 10 (test) is loaded ONLY by evaluate_final_models() AFTER tuning is frozen.
"""

import json
import sys
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from ml_config import (
    FEATURE_CACHE_DIR, SAVED_MODELS_DIR, RESULTS_DIR,
    CLASS_NAMES, NUM_CLASSES, SVM_SEED, FEATURE_CONFIG_HASH
)

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cache loader with integrity check
# ---------------------------------------------------------------------------
def load_cache(path: Path) -> dict:
    d = dict(np.load(path, allow_pickle=True))
    stored_hash = str(d["feature_config_hash"].item())
    if stored_hash != FEATURE_CONFIG_HASH:
        raise ValueError(
            f"Feature config hash mismatch in {path}.\n"
            f"  Expected: {FEATURE_CONFIG_HASH}\n"
            f"  Got:      {stored_hash}"
        )
    return d


# ---------------------------------------------------------------------------
# Tuning function (D15: accepts ONLY train+val; never receives test data)
# ---------------------------------------------------------------------------
def tune_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: list,
) -> Tuple[dict, float, list]:
    """
    Grid search over param_grid using validation accuracy.
    Fold 10 (test) data is never passed to this function.

    Returns
    -------
    best_params : dict
    best_val_acc : float
    results_list : list of dicts with all grid results sorted by val_accuracy
    """
    rows = []
    best_val_acc = -1.0
    best_params = {}

    for params in param_grid:
        C = params["C"]
        gamma = params["gamma"]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=C, gamma=gamma,
                        probability=False, random_state=SVM_SEED))
        ])
        pipeline.fit(X_train, y_train)
        val_acc = float((pipeline.predict(X_val) == y_val).mean())

        rows.append({"C": C, "gamma": gamma, "val_accuracy": val_acc})
        print(f"  C={C}, gamma={gamma} → val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params

    results_sorted = sorted(rows, key=lambda r: r["val_accuracy"], reverse=True)
    return best_params, best_val_acc, results_sorted


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------
def train_svm_model(variant: str):
    """
    variant: 'original' or 'augmented'
    """
    assert variant in ("original", "augmented"), f"Unknown variant: {variant}"

    print(f"\n{'=' * 60}")
    print(f"SVM TRAINING — {variant.upper()}")
    print(f"{'=' * 60}")

    # Load train features
    train_cache = (
        FEATURE_CACHE_DIR / "features_train_original.npz"
        if variant == "original"
        else FEATURE_CACHE_DIR / "features_train_augmented.npz"
    )
    val_cache  = FEATURE_CACHE_DIR / "features_val.npz"

    # NOTE: test cache is loaded ONLY in evaluate_final_models(), NEVER here.
    train_data = load_cache(train_cache)
    val_data   = load_cache(val_cache)

    X_train = train_data["features"].astype(np.float64)
    y_train = train_data["labels"].astype(np.int32)
    X_val   = val_data["features"].astype(np.float64)
    y_val   = val_data["labels"].astype(np.int32)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Empirical pilot: estimate fit time on subsets
    print("\n[Pilot] Estimating SVM fit time on subsets...")
    for frac in [0.25, 0.50, 1.0]:
        n = max(1, int(len(X_train) * frac))
        idx = np.random.default_rng(42).choice(len(X_train), n, replace=False)
        scaler = StandardScaler()
        Xp = scaler.fit_transform(X_train[idx])
        t0 = time.perf_counter()
        SVC(kernel="rbf", C=1.0, gamma="scale",
            probability=False, random_state=SVM_SEED).fit(Xp, y_train[idx])
        elapsed = time.perf_counter() - t0
        print(f"  n={n:5d} ({frac*100:.0f}%): {elapsed:.1f} s")

    # Stage 1: coarse grid
    print("\n[Stage 1] Coarse grid search...")
    stage1_grid = [
        {"C": C, "gamma": g}
        for C in [0.1, 1.0, 10.0]
        for g in ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
    ]  # 18 fits
    best_params, best_val_acc, df1 = tune_svm(
        X_train, y_train, X_val, y_val, stage1_grid
    )
    print(f"\nStage 1 best: {best_params}  val_acc={best_val_acc:.4f}")

    # Stage 2: fine grid around best
    best_C = best_params["C"]
    if isinstance(best_params["gamma"], float):
        best_g = best_params["gamma"]
        fine_gammas = [best_g * f for f in [0.3, 1.0, 3.0]]
    else:
        fine_gammas = [best_params["gamma"]]

    stage2_grid = [
        {"C": C, "gamma": g}
        for C in [best_C / 3, best_C, best_C * 3]
        for g in fine_gammas
    ]  # up to 9 fits
    best_params, best_val_acc, df2 = tune_svm(
        X_train, y_train, X_val, y_val, stage2_grid
    )
    print(f"\nStage 2 best: {best_params}  val_acc={best_val_acc:.4f}")

    # Final fit on full training set with best hyperparameters
    print("\n[Final fit] Training final SVM on full training set...")
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf",
                    C=best_params["C"],
                    gamma=best_params["gamma"],
                    probability=False,
                    random_state=SVM_SEED))
    ])
    t0 = time.perf_counter()
    final_pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    val_acc_final = (final_pipeline.predict(X_val) == y_val).mean()
    print(f"Final val_accuracy = {val_acc_final:.4f}  (fit time: {fit_time:.1f} s)")

    # Save model and scaler separately
    model_path  = SAVED_MODELS_DIR / f"svm_{variant}.joblib"
    scaler_path = SAVED_MODELS_DIR / f"svm_{variant}_scaler.joblib"
    joblib.dump(final_pipeline, model_path)
    joblib.dump(final_pipeline.named_steps["scaler"], scaler_path)
    print(f"Saved: {model_path}")
    print(f"Saved: {scaler_path}")

    # Save tuning results
    tuning_path = RESULTS_DIR / f"svm_{variant}_tuning.json"
    tuning_results = {
        "variant": variant,
        "best_params": {
            "C": float(best_params["C"]),
            "gamma": str(best_params["gamma"]),
        },
        "val_accuracy": float(val_acc_final),
        "fit_time_s": round(fit_time, 2),
        "stage1_results": df1,
        "stage2_results": df2,
    }
    with open(tuning_path, "w") as f:
        json.dump(tuning_results, f, indent=2)
    print(f"Saved: {tuning_path}")

    return final_pipeline, val_acc_final


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["original", "augmented", "both"],
                   default="both")
    args = p.parse_args()

    if args.variant in ("original", "both"):
        train_svm_model("original")
    if args.variant in ("augmented", "both"):
        train_svm_model("augmented")

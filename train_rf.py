"""
train_rf.py
Phase F: Train Random Forest classifiers (original and augmented features).

Runs in .venv_ml. Zero dependency on TensorFlow.
Random Forest uses RAW (unscaled) features — no StandardScaler.
Fold 10 (test) is loaded ONLY by evaluate_final_models() AFTER tuning is frozen.
"""

import json
import sys
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from ml_config import (
    FEATURE_CACHE_DIR, SAVED_MODELS_DIR, RESULTS_DIR,
    CLASS_NAMES, NUM_CLASSES, RF_SEED, FEATURE_CONFIG_HASH
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
def tune_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: list,
) -> Tuple[dict, float, list]:
    """
    Grid search over param_grid using validation accuracy.
    Fold 10 (test) data is never passed to this function.
    Uses n_jobs=-1 during exploration for speed; final refit uses n_jobs=1.

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
        n_est   = params["n_estimators"]
        max_dep = params["max_depth"]
        min_sl  = params["min_samples_leaf"]

        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_dep,
            min_samples_leaf=min_sl,
            random_state=RF_SEED,
            n_jobs=-1,   # parallel during tuning (exploratory only)
        )
        rf.fit(X_train, y_train)
        val_acc = float((rf.predict(X_val) == y_val).mean())

        rows.append({
            "n_estimators": n_est,
            "max_depth": max_dep,
            "min_samples_leaf": min_sl,
            "val_accuracy": val_acc,
        })
        print(f"  n_est={n_est}, max_depth={max_dep}, min_sl={min_sl} → val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params  = params

    results_sorted = sorted(rows, key=lambda r: r["val_accuracy"], reverse=True)
    return best_params, best_val_acc, results_sorted


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------
def train_rf_model(variant: str):
    """
    variant: 'original' or 'augmented'
    Random Forest uses unscaled features (no StandardScaler).
    """
    assert variant in ("original", "augmented"), f"Unknown variant: {variant}"

    print(f"\n{'=' * 60}")
    print(f"RANDOM FOREST TRAINING — {variant.upper()}")
    print(f"{'=' * 60}")

    # Load train features
    train_cache = (
        FEATURE_CACHE_DIR / "features_train_original.npz"
        if variant == "original"
        else FEATURE_CACHE_DIR / "features_train_augmented.npz"
    )
    val_cache = FEATURE_CACHE_DIR / "features_val.npz"

    # NOTE: test cache is loaded ONLY in evaluate_final_models(), NEVER here.
    train_data = load_cache(train_cache)
    val_data   = load_cache(val_cache)

    X_train = train_data["features"].astype(np.float32)
    y_train = train_data["labels"].astype(np.int32)
    X_val   = val_data["features"].astype(np.float32)
    y_val   = val_data["labels"].astype(np.int32)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print("Note: Random Forest uses RAW unscaled features (no StandardScaler).")

    # Grid search (72 fits across two stages)
    print("\n[Stage 1] Coarse grid (n_estimators × max_depth × min_samples_leaf)...")
    stage1_grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m}
        for n in [100, 300, 500]
        for d in [None, 20, 40]
        for m in [1, 2, 4]
    ]  # 3×3×3 = 27 fits
    best_params, best_val_acc, df1 = tune_rf(
        X_train, y_train, X_val, y_val, stage1_grid
    )
    print(f"\nStage 1 best: {best_params}  val_acc={best_val_acc:.4f}")

    # Stage 2: refine around best
    best_n   = best_params["n_estimators"]
    best_d   = best_params["max_depth"]
    best_m   = best_params["min_samples_leaf"]

    n_range = sorted(set([max(50, best_n - 100), best_n, best_n + 100]))
    if best_d is None:
        d_range = [None, 30, 50]
    else:
        d_range = sorted(set([max(10, best_d - 10), best_d, best_d + 10]))
    m_range = sorted(set([max(1, best_m - 1), best_m, best_m + 1]))

    stage2_grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m}
        for n in n_range
        for d in d_range
        for m in m_range
    ]
    print(f"\n[Stage 2] Fine grid ({len(stage2_grid)} fits)...")
    best_params, best_val_acc, df2 = tune_rf(
        X_train, y_train, X_val, y_val, stage2_grid
    )
    print(f"\nStage 2 best: {best_params}  val_acc={best_val_acc:.4f}")

    # Final refit with n_jobs=1 for full reproducibility
    print("\n[Final fit] Training final RF with n_jobs=1 (deterministic)...")
    final_rf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=RF_SEED,
        n_jobs=1,   # deterministic
    )
    t0 = time.perf_counter()
    final_rf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    val_acc_final = (final_rf.predict(X_val) == y_val).mean()
    print(f"Final val_accuracy = {val_acc_final:.4f}  (fit time: {fit_time:.1f} s)")

    # Verify: RF is a bare RandomForestClassifier, NOT a Pipeline (D14 check)
    assert isinstance(final_rf, RandomForestClassifier), (
        "RF model must be RandomForestClassifier, not Pipeline!"
    )

    # Save model
    model_path = SAVED_MODELS_DIR / f"rf_{variant}.joblib"
    joblib.dump(final_rf, model_path)
    print(f"Saved: {model_path}")

    # Save tuning results
    tuning_path = RESULTS_DIR / f"rf_{variant}_tuning.json"
    tuning_results = {
        "variant": variant,
        "best_params": {
            "n_estimators": int(best_params["n_estimators"]),
            "max_depth": best_params["max_depth"],
            "min_samples_leaf": int(best_params["min_samples_leaf"]),
        },
        "val_accuracy": float(val_acc_final),
        "fit_time_s": round(fit_time, 2),
        "stage1_results": df1,
        "stage2_results": df2,
    }
    with open(tuning_path, "w") as f:
        json.dump(tuning_results, f, indent=2)
    print(f"Saved: {tuning_path}")

    return final_rf, val_acc_final


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["original", "augmented", "both"],
                   default="both")
    args = p.parse_args()

    if args.variant in ("original", "both"):
        train_rf_model("original")
    if args.variant in ("augmented", "both"):
        train_rf_model("augmented")

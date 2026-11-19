"""
build_feature_cache.py
Phase E: Builds all feature cache .npz files and noisy waveform .npy caches.

Runs in .venv (for waveform generation) then .venv_ml (for feature extraction).
This script handles BOTH steps in sequence:
  Step 1 (.venv)    — Load, preprocess, and cache clean waveforms + noisy waveforms.
  Step 2 (.venv_ml) — Extract 250-dim features from cached waveforms.

Usage (run from project root):
  # Step 1: generate waveform caches (run with .venv)
  .\.venv\Scripts\python.exe build_feature_cache.py --step waveforms

  # Step 2: extract features from waveform caches (run with .venv_ml)
  .\.venv_ml\Scripts\python.exe build_feature_cache.py --step features

Each step is idempotent: re-running skips existing outputs unless --force is passed.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing (must be done before heavy imports to keep steps isolated)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Build feature/waveform caches.")
parser.add_argument(
    "--step",
    choices=["waveforms", "features"],
    required=True,
    help="Which step to run."
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite existing cache files."
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Project root on sys.path so local modules resolve
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from ml_config import (
    DATASET_ROOT, METADATA_PATH, AUDIO_ROOT,
    FEATURE_CACHE_DIR, TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS,
    SAMPLE_RATE, AUGMENT_SEED, NOISE_SEED, SNR_LEVELS,
    WAVEFORM_CACHE_TEMPLATE, NAN_REPORT_PATH,
    CACHE_TRAIN_ORIGINAL, CACHE_TRAIN_AUGMENTED, CACHE_VAL, CACHE_TEST,
    FEATURE_CONFIG_HASH, make_seed
)

FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 1: Waveform caching (must run in .venv — uses preprocessing.py)
# ---------------------------------------------------------------------------
if args.step == "waveforms":

    # Guard: these imports require scipy + soundfile (in .venv), not librosa
    import pandas as pd
    from preprocessing import (
        load_metadata, create_data_splits,
        get_audio_path, load_and_preprocess_audio
    )
    from augmentation import add_noise
    from training_augmentation import augment_audio

    print("=" * 60)
    print("STEP 1: WAVEFORM CACHING (.venv)")
    print("=" * 60)

    # -- Load metadata and splits --
    if not METADATA_PATH.is_file():
        print(f"[ERROR] Metadata not found: {METADATA_PATH}")
        print("Set URBANSOUND8K_PATH env var to the dataset root.")
        sys.exit(1)

    meta = load_metadata(METADATA_PATH)
    train_df, val_df, test_df = create_data_splits(meta)
    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    def load_waveforms(df, label):
        """Load and preprocess waveforms for a dataframe; return (waveforms, filenames, folds, labels)."""
        waveforms, filenames, folds, labels = [], [], [], []
        errors = 0
        for _, row in df.iterrows():
            p = get_audio_path(AUDIO_ROOT, row["fold"], row["slice_file_name"])
            try:
                wav = load_and_preprocess_audio(p)   # shape (88200, 1)
                waveforms.append(wav.squeeze())       # (88200,)
                filenames.append(str(row["slice_file_name"]))
                folds.append(int(row["fold"]))
                labels.append(int(row["classID"]))
            except Exception as exc:
                print(f"  [WARN] {label}: skip {row['slice_file_name']}: {exc}")
                errors += 1
        print(f"  Loaded {len(waveforms)} waveforms ({errors} errors) for {label}")
        return (
            np.array(waveforms, dtype=np.float32),
            np.array(filenames),
            np.array(folds, dtype=np.int32),
            np.array(labels, dtype=np.int32),
        )

    # -- Clean waveform caches --
    clean_cache = FEATURE_CACHE_DIR / "clean_waveforms_train.npy"
    clean_meta  = FEATURE_CACHE_DIR / "clean_meta_train.npz"
    val_cache   = FEATURE_CACHE_DIR / "clean_waveforms_val.npy"
    val_meta    = FEATURE_CACHE_DIR / "clean_meta_val.npz"
    test_cache  = FEATURE_CACHE_DIR / "clean_waveforms_test.npy"
    test_meta   = FEATURE_CACHE_DIR / "clean_meta_test.npz"

    if not clean_cache.exists() or args.force:
        print("\n[1/5] Loading training waveforms (folds 1-8)...")
        X_tr, fn_tr, fo_tr, y_tr = load_waveforms(train_df, "train")
        np.save(clean_cache, X_tr)
        np.savez(clean_meta, filenames=fn_tr, folds=fo_tr, labels=y_tr)
        print(f"  Saved: {clean_cache}  shape={X_tr.shape}")
    else:
        print(f"[1/5] Training waveforms already cached: {clean_cache} (use --force to rebuild)")
        X_tr = np.load(clean_cache)
        m = np.load(clean_meta)
        fn_tr, fo_tr, y_tr = m["filenames"], m["folds"], m["labels"]

    if not val_cache.exists() or args.force:
        print("\n[2/5] Loading validation waveforms (fold 9)...")
        X_val, fn_val, fo_val, y_val = load_waveforms(val_df, "val")
        np.save(val_cache, X_val)
        np.savez(val_meta, filenames=fn_val, folds=fo_val, labels=y_val)
        print(f"  Saved: {val_cache}  shape={X_val.shape}")
    else:
        print(f"[2/5] Validation waveforms already cached: {val_cache}")
        X_val = np.load(val_cache)
        m = np.load(val_meta)
        fn_val, fo_val, y_val = m["filenames"], m["folds"], m["labels"]

    if not test_cache.exists() or args.force:
        print("\n[3/5] Loading test waveforms (fold 10)...")
        X_te, fn_te, fo_te, y_te = load_waveforms(test_df, "test")
        np.save(test_cache, X_te)
        np.savez(test_meta, filenames=fn_te, folds=fo_te, labels=y_te)
        print(f"  Saved: {test_cache}  shape={X_te.shape}")
    else:
        print(f"[3/5] Test waveforms already cached: {test_cache}")
        X_te = np.load(test_cache)
        m = np.load(test_meta)
        fn_te, fo_te, y_te = m["filenames"], m["folds"], m["labels"]

    # -- Augmented waveform cache (training only, deterministic per clip) --
    aug_cache = FEATURE_CACHE_DIR / "aug_waveforms_train.npy"
    if not aug_cache.exists() or args.force:
        print("\n[4/5] Generating augmented training waveforms (deterministic)...")
        X_aug = np.zeros_like(X_tr)
        for i, (wav, fname, fold) in enumerate(zip(X_tr, fn_tr, fo_tr)):
            seed = make_seed(AUGMENT_SEED, "aug", int(fold), fname, i)
            prev_state = np.random.get_state()
            try:
                np.random.seed(seed)
                aug = augment_audio(wav.reshape(-1, 1))  # expects (N,1)
                X_aug[i] = aug.squeeze()
            finally:
                np.random.set_state(prev_state)
            if (i + 1) % 500 == 0:
                print(f"  Augmented {i+1}/{len(X_tr)} clips...")
        np.save(aug_cache, X_aug)
        print(f"  Saved: {aug_cache}  shape={X_aug.shape}")
    else:
        print(f"[4/5] Augmented waveform cache already exists: {aug_cache}")

    # -- Noisy waveform caches (test set, per SNR) --
    print("\n[5/5] Generating noisy test waveforms...")
    for snr in SNR_LEVELS:
        out_path = Path(WAVEFORM_CACHE_TEMPLATE.format(snr=snr))
        if out_path.exists() and not args.force:
            print(f"  SNR={snr} dB: already cached → {out_path}")
            continue
        X_noisy = np.zeros_like(X_te)
        for i, (wav, fname, fold) in enumerate(zip(X_te, fn_te, fo_te)):
            seed = make_seed(NOISE_SEED, "noise", int(fold), fname, snr)
            prev_state = np.random.get_state()
            try:
                np.random.seed(seed)
                noisy = add_noise(wav, snr_db=snr)
            finally:
                np.random.set_state(prev_state)
            X_noisy[i] = noisy
        np.save(out_path, X_noisy)
        print(f"  Saved: {out_path}  shape={X_noisy.shape}")

    print("\n[STEP 1 COMPLETE] All waveform caches ready.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# STEP 2: Feature extraction (must run in .venv_ml — uses librosa)
# ---------------------------------------------------------------------------
if args.step == "features":

    from tqdm import tqdm
    from feature_extraction import extract_features

    print("=" * 60)
    print("STEP 2: FEATURE EXTRACTION (.venv_ml)")
    print("=" * 60)

    def extract_batch(waveforms, filenames, folds, labels, aug_type, cache_path, desc):
        """Extract features for a batch of waveforms; impute non-finite using train mean."""
        if cache_path.exists() and not args.force:
            print(f"  Already exists: {cache_path} (use --force to rebuild)")
            return

        N = len(waveforms)
        feat_matrix = np.zeros((N, 250), dtype=np.float32)
        nan_clips = []

        for i in tqdm(range(N), desc=desc):
            try:
                f = extract_features(waveforms[i])
            except Exception as exc:
                print(f"\n  [ERROR] Extraction failed for clip {i} ({filenames[i]}): {exc}")
                f = np.full(250, np.nan, dtype=np.float32)
                nan_clips.append({"index": i, "file": str(filenames[i]), "reason": str(exc)})

            if not np.isfinite(f).all():
                bad_dims = np.where(~np.isfinite(f))[0].tolist()
                nan_clips.append({
                    "index": i, "file": str(filenames[i]),
                    "bad_dims": bad_dims
                })

            feat_matrix[i] = f

        return feat_matrix, nan_clips

    # Load waveform caches (produced by step 1)
    def _load(wav_path, meta_path):
        X = np.load(wav_path)
        m = np.load(meta_path)
        return X, m["filenames"], m["folds"], m["labels"]

    cache_dir = FEATURE_CACHE_DIR
    X_tr, fn_tr, fo_tr, y_tr = _load(
        cache_dir / "clean_waveforms_train.npy",
        cache_dir / "clean_meta_train.npz"
    )
    X_val, fn_val, fo_val, y_val = _load(
        cache_dir / "clean_waveforms_val.npy",
        cache_dir / "clean_meta_val.npz"
    )
    X_te, fn_te, fo_te, y_te = _load(
        cache_dir / "clean_waveforms_test.npy",
        cache_dir / "clean_meta_test.npz"
    )
    X_aug = np.load(cache_dir / "aug_waveforms_train.npy")

    all_nan_clips = []

    # --- Train original ---
    if not CACHE_TRAIN_ORIGINAL.exists() or args.force:
        print("\n[1/4] Extracting train-original features...")
        feat_tr, nans = extract_batch(X_tr, fn_tr, fo_tr, y_tr, "original",
                                      CACHE_TRAIN_ORIGINAL, "train-original")
        all_nan_clips.extend(nans)
    else:
        print(f"\n[1/4] Already exists: {CACHE_TRAIN_ORIGINAL}")
        d = np.load(CACHE_TRAIN_ORIGINAL)
        feat_tr = d["features"]

    # Compute training mean for imputation (finite values only)
    # This is computed from clean train features and applied uniformly
    finite_mask = np.isfinite(feat_tr)
    train_mean = np.where(
        finite_mask.any(axis=0),
        np.nanmean(feat_tr, axis=0),
        0.0
    ).astype(np.float32)

    def impute_and_save(feat, nans, nan_label, filenames, folds, labels, aug_type, cache_path):
        """Impute non-finite values with train_mean and save cache."""
        if cache_path.exists() and not args.force:
            return

        # Imputation
        for i in range(len(feat)):
            bad = ~np.isfinite(feat[i])
            if bad.any():
                feat[i][bad] = train_mean[bad]
                print(f"  Imputed {bad.sum()} dims in clip {i} ({filenames[i]}) [{nan_label}]")

        # Final finite check
        assert np.isfinite(feat).all(), "Non-finite values remain after imputation!"

        np.savez(
            cache_path,
            features=feat,
            labels=labels,
            filenames=filenames,
            folds=folds,
            aug_type=np.array([aug_type] * len(feat)),
            feature_config_hash=np.array([FEATURE_CONFIG_HASH])
        )
        print(f"  Saved: {cache_path}  shape={feat.shape}")

    # Save train original
    impute_and_save(feat_tr, all_nan_clips, "train-orig",
                    fn_tr, fo_tr, y_tr, "original", CACHE_TRAIN_ORIGINAL)

    # --- Train augmented ---
    if not CACHE_TRAIN_AUGMENTED.exists() or args.force:
        print("\n[2/4] Extracting train-augmented features...")
        feat_aug, nans = extract_batch(X_aug, fn_tr, fo_tr, y_tr, "augmented",
                                       CACHE_TRAIN_AUGMENTED, "train-augmented")
        all_nan_clips.extend(nans)
        impute_and_save(feat_aug, nans, "train-aug",
                        fn_tr, fo_tr, y_tr, "augmented", CACHE_TRAIN_AUGMENTED)
    else:
        print(f"\n[2/4] Already exists: {CACHE_TRAIN_AUGMENTED}")

    # --- Validation ---
    if not CACHE_VAL.exists() or args.force:
        print("\n[3/4] Extracting validation features...")
        feat_val, nans = extract_batch(X_val, fn_val, fo_val, y_val, "original",
                                       CACHE_VAL, "val")
        all_nan_clips.extend(nans)
        impute_and_save(feat_val, nans, "val",
                        fn_val, fo_val, y_val, "original", CACHE_VAL)
    else:
        print(f"\n[3/4] Already exists: {CACHE_VAL}")

    # --- Test ---
    if not CACHE_TEST.exists() or args.force:
        print("\n[4/4] Extracting test features...")
        feat_te, nans = extract_batch(X_te, fn_te, fo_te, y_te, "original",
                                      CACHE_TEST, "test")
        all_nan_clips.extend(nans)
        impute_and_save(feat_te, nans, "test",
                        fn_te, fo_te, y_te, "original", CACHE_TEST)
    else:
        print(f"\n[4/4] Already exists: {CACHE_TEST}")

    # --- Save NaN report ---
    if all_nan_clips:
        with open(NAN_REPORT_PATH, "w") as f:
            json.dump(all_nan_clips, f, indent=2)
        print(f"\n[WARN] {len(all_nan_clips)} clips had non-finite features → {NAN_REPORT_PATH}")
    else:
        print("\n[OK] No non-finite features detected.")

    print("\n[STEP 2 COMPLETE] All feature caches ready.")
    sys.exit(0)

"""
Training pipeline for the Environmental Sound Classification project.

Trains three architectures on UrbanSound8K (22050 Hz, 4 s, 10 classes):

1. Baseline 1D CNN       → saved_models/baseline_cnn.keras
2. Hybrid 1D CNN-BiGRU   → saved_models/cnn_bigru.keras
3. Paper-style 1D CNN    → saved_models/paper_cnn.keras

Dataset split:
    Train      : folds 1-8
    Validation : fold 9
    Test       : fold 10

Usage:
    python train.py
    python train.py --model baseline
    python train.py --model bigru
    python train.py --model paper
    python train.py --epochs 30 --batch 16
    python train.py --augment
    python train.py --smoke-test
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path


# Fix Matplotlib cache warning on Windows
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_ROOT = Path(
    os.environ.get(
        "URBANSOUND8K_PATH",
        PROJECT_DIR.parent / "archive"
    )
)

METADATA_PATH = DATASET_ROOT / "UrbanSound8K.csv"
AUDIO_ROOT = DATASET_ROOT

SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"
RESULTS_DIR = PROJECT_DIR / "results"

BASELINE_MODEL_PATH = SAVED_MODELS_DIR / "baseline_cnn.keras"
BIGRU_MODEL_PATH = SAVED_MODELS_DIR / "cnn_bigru.keras"
PAPER_MODEL_PATH = SAVED_MODELS_DIR / "paper_cnn.keras"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.001

NUM_CLASSES = 10
INPUT_SHAPE = (88200, 1)

TRAIN_FOLDS = list(range(1, 9))
VAL_FOLDS = [9]
TEST_FOLDS = [10]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        print(
            "\n[ERROR] TensorFlow is not installed in this environment.\n"
            "Install it with:\n"
            "    pip install tensorflow\n"
        )
        sys.exit(1)


def _make_tf_dataset(
    metadata_df,
    audio_root,
    batch_size,
    shuffle,
    augment=False
):
    """Create a streaming tf.data.Dataset."""

    tf = _require_tensorflow()

    if augment:
        from training_augmentation import augment_audio

    n_samples = len(metadata_df)

    def _gen():
        import numpy as np
        from preprocessing import (
            get_audio_path,
            load_and_preprocess_audio
        )

        rng = np.random.default_rng() if shuffle else None

        indices = (
            rng.permutation(n_samples)
            if shuffle
            else np.arange(n_samples)
        )

        for idx in indices:
            row = metadata_df.iloc[int(idx)]

            audio_path = get_audio_path(
                audio_root,
                row["fold"],
                row["slice_file_name"]
            )

            try:
                processed = load_and_preprocess_audio(audio_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load or preprocess audio file "
                    f"'{row['slice_file_name']}' "
                    f"in fold {row['fold']} from {audio_path}: {exc}"
                ) from exc

            if augment:
                processed = augment_audio(processed)

            yield (
                processed.astype(np.float32),
                np.int64(row["classID"])
            )

    dataset = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(
                shape=INPUT_SHAPE,
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.int64
            ),
        ),
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(256, n_samples)
        )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, n_samples


def _build_callbacks(model_path, monitor="val_accuracy"):
    """Return standard training callbacks."""

    tf = _require_tensorflow()

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def _save_history(history, name):
    """Save Keras history dict."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    hist_path = RESULTS_DIR / f"{name}_history.json"

    hist_dict = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }

    with open(
        hist_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            hist_dict,
            f,
            indent=2
        )

    print(f"  History saved → {hist_path}")

    return hist_path


def _save_training_meta(
    name,
    model_path,
    history,
    elapsed_sec,
    batch_size,
    epochs_run
):
    """Persist training metadata."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    meta = {
        "model_name": name,
        "model_path": str(model_path),
        "training_time_sec": round(
            elapsed_sec,
            2
        ),
        "epochs_run": epochs_run,
        "best_val_accuracy": max(
            history.history.get(
                "val_accuracy",
                [0]
            )
        ),
        "best_val_loss": min(
            history.history.get(
                "val_loss",
                [float("inf")]
            )
        ),
        "batch_size": batch_size,
    }

    meta_path = RESULTS_DIR / f"{name}_training_meta.json"

    with open(
        meta_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            meta,
            f,
            indent=2
        )

    print(f"  Training meta saved → {meta_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model_name,
    model_builder,
    model_path,
    epochs,
    batch_size,
    augment,
    subset_size=None
):
    """Full training loop for one model."""

    tf = _require_tensorflow()

    print(f"\n{'=' * 60}")
    print(f"  Training: {model_name}")
    print(f"{'=' * 60}")

    print(f"  Dataset   : {DATASET_ROOT}")
    print(f"  Metadata  : {METADATA_PATH}")
    print(f"  Audio root: {AUDIO_ROOT}")
    print(
        f"  Epochs    : {epochs} | "
        f"Batch size: {batch_size} | "
        f"Augment: {augment}"
    )
    print(f"  Checkpoint: {model_path}")

    # Validate paths
    if not METADATA_PATH.is_file():
        print(
            f"\n[ERROR] Metadata not found: "
            f"{METADATA_PATH}"
        )
        sys.exit(1)

    if not AUDIO_ROOT.is_dir():
        print(
            f"\n[ERROR] Audio directory not found: "
            f"{AUDIO_ROOT}"
        )
        sys.exit(1)

    # Load metadata and split
    print(
        "\n  Loading metadata and creating splits..."
    )

    from preprocessing import (
        load_metadata,
        create_data_splits,
        validate_dataset
    )

    metadata = load_metadata(
        METADATA_PATH
    )

    val_results = validate_dataset(
        metadata,
        AUDIO_ROOT
    )

    if not val_results["all_folds_exist"]:
        print(
            f"[ERROR] Missing fold directories: "
            f"{val_results['missing_folds']}"
        )
        sys.exit(1)

    if val_results["missing_files"]:
        n_miss = len(
            val_results["missing_files"]
        )

        print(
            f"[ERROR] {n_miss} audio file(s) "
            "missing from dataset."
        )
        sys.exit(1)

    train_df, val_df, _ = create_data_splits(
        metadata
    )

    if subset_size is not None and subset_size > 0:
        print(
            f"  [SMOKE-TEST] Subsetting train "
            f"data to {subset_size} samples and "
            f"val to {min(subset_size, len(val_df))}"
        )

        train_df = (
            train_df
            .head(subset_size)
            .copy()
            .reset_index(drop=True)
        )

        val_df = (
            val_df
            .head(min(subset_size, len(val_df)))
            .copy()
            .reset_index(drop=True)
        )

    print(
        f"  Train samples : {len(train_df)}"
    )

    print(
        f"  Val samples   : {len(val_df)}"
    )

    # Build model
    print("\n  Building model...")

    model = model_builder()

    model.summary(
        line_length=80
    )

    # Build datasets
    print("\n  Building data pipelines...")

    train_ds, n_train = _make_tf_dataset(
        train_df,
        AUDIO_ROOT,
        batch_size,
        shuffle=True,
        augment=augment
    )

    val_ds, n_val = _make_tf_dataset(
        val_df,
        AUDIO_ROOT,
        batch_size,
        shuffle=False,
        augment=False
    )

    # Repeat so explicit steps control epoch boundaries
    train_ds = train_ds.repeat()
    val_ds = val_ds.repeat()

    steps_per_epoch = math.ceil(
        n_train / batch_size
    )

    # Avoid evaluating duplicated validation samples
    validation_steps = max(
        1,
        n_val // batch_size
    )

    print(
        f"\n  Steps per epoch      : "
        f"{steps_per_epoch}"
    )

    print(
        f"  Validation steps     : "
        f"{validation_steps}"
    )

    # Callbacks
    SAVED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    callbacks = _build_callbacks(
        model_path
    )

    # Train
    print(
        f"\n  Starting training "
        f"({steps_per_epoch} training steps, "
        f"{validation_steps} validation steps)...\n"
    )

    t_start = time.time()

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - t_start

    epochs_run = len(
        history.history["loss"]
    )

    best_val_acc = max(
        history.history.get(
            "val_accuracy",
            [0.0]
        )
    )

    print(
        f"\n  {'─' * 50}"
    )

    print(
        f"  Training complete in "
        f"{elapsed:.1f} s "
        f"({elapsed / 60:.1f} min)"
    )

    print(
        f"  Epochs run      : {epochs_run}"
    )

    print(
        f"  Best val acc    : "
        f"{best_val_acc:.4f}"
    )

    if model_path.is_file():
        print(
            f"  Best model saved: "
            f"{model_path}"
        )
    else:
        model.save(
            str(model_path)
        )

        print(
            f"  Model saved: "
            f"{model_path}"
        )

    _save_history(
        history,
        model_name
    )

    _save_training_meta(
        model_name,
        model_path,
        history,
        elapsed,
        batch_size,
        epochs_run
    )

    return history, elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train Baseline 1D CNN, "
            "Hybrid CNN-BiGRU, and "
            "Paper-style 1D CNN "
            "on UrbanSound8K."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "baseline",
            "bigru",
            "paper",
            "both"
        ],
        default="both",
        help=(
            "Which model to train."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=(
            f"Maximum training epochs "
            f"(default: {DEFAULT_EPOCHS})."
        ),
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"Batch size "
            f"(default: {DEFAULT_BATCH_SIZE})."
        ),
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Apply training-time augmentation "
            "(noise, volume, time-shift)."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=(
            f"Initial learning rate "
            f"(default: {DEFAULT_LEARNING_RATE})."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run lightweight smoke test "
            "on real dataset."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = parse_args()

    tf = _require_tensorflow()

    print(
        f"TensorFlow {tf.__version__}"
    )

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        f"GPUs available: "
        f"{gpus if gpus else 'None (CPU only)'}"
    )

    from models import (
        build_baseline_cnn,
        build_cnn_bigru,
        build_paper_cnn
    )

    # -------------------------------------------------------
    # Model builders with optional learning-rate override
    # -------------------------------------------------------

    if args.lr != DEFAULT_LEARNING_RATE:

        def _baseline_with_lr():

            m = build_baseline_cnn()

            m.optimizer.learning_rate.assign(args.lr)

            return m

        def _bigru_with_lr():
            m = build_cnn_bigru()
            m.optimizer.learning_rate.assign(args.lr)
            return m

        def _paper_with_lr():

            m = build_paper_cnn()

            tf.keras.backend.set_value(
                m.optimizer.learning_rate,
                args.lr
            )

            return m

        baseline_builder = _baseline_with_lr
        bigru_builder = _bigru_with_lr
        paper_builder = _paper_with_lr

    else:

        baseline_builder = build_baseline_cnn
        bigru_builder = build_cnn_bigru
        paper_builder = build_paper_cnn

    # -------------------------------------------------------
    # Model selection
    # -------------------------------------------------------

    train_baseline = args.model in (
        "baseline",
        "both"
    )

    train_bigru = args.model in (
        "bigru",
        "both"
    )

    train_paper = args.model in (
        "paper",
        "both"
    )

    # -------------------------------------------------------
    # Training configuration
    # -------------------------------------------------------

    epochs = (
        1
        if args.smoke_test
        else args.epochs
    )

    batch_size = (
        4
        if args.smoke_test
        else args.batch
    )

    subset_size = (
        16
        if args.smoke_test
        else None
    )

    # -------------------------------------------------------
    # Model paths
    # -------------------------------------------------------

    if args.smoke_test:

        baseline_path = (
            SAVED_MODELS_DIR /
            "smoke_baseline.keras"
        )

        bigru_path = (
            SAVED_MODELS_DIR /
            "smoke_bigru.keras"
        )

        paper_path = (
            SAVED_MODELS_DIR /
            "smoke_paper.keras"
        )

        print(
            "\n[SMOKE-TEST MODE ACTIVE]"
        )

        print(
            "Running lightweight "
            "1-epoch real-audio validation..."
        )

    else:

        baseline_path = (
            BASELINE_MODEL_PATH
        )

        bigru_path = (
            BIGRU_MODEL_PATH
        )

        paper_path = (
            PAPER_MODEL_PATH
        )

    summaries = []

    # -------------------------------------------------------
    # Baseline CNN
    # -------------------------------------------------------

    if train_baseline:

        history, elapsed = train_model(

            model_name=(
                "baseline_cnn"
                if not args.smoke_test
                else "smoke_baseline"
            ),

            model_builder=baseline_builder,

            model_path=baseline_path,

            epochs=epochs,

            batch_size=batch_size,

            augment=args.augment,

            subset_size=subset_size,
        )

        summaries.append({

            "model":
                "Baseline 1D CNN",

            "saved_to":
                str(baseline_path),

            "best_val_acc":
                max(
                    history.history.get(
                        "val_accuracy",
                        [0.0]
                    )
                ),

            "time_sec":
                round(
                    elapsed,
                    1
                ),
        })

    # -------------------------------------------------------
    # Hybrid CNN-BiGRU
    # -------------------------------------------------------

    if train_bigru:

        history, elapsed = train_model(

            model_name=(
                "cnn_bigru"
                if not args.smoke_test
                else "smoke_bigru"
            ),

            model_builder=bigru_builder,

            model_path=bigru_path,

            epochs=epochs,

            batch_size=batch_size,

            augment=args.augment,

            subset_size=subset_size,
        )

        summaries.append({

            "model":
                "Hybrid CNN-BiGRU",

            "saved_to":
                str(bigru_path),

            "best_val_acc":
                max(
                    history.history.get(
                        "val_accuracy",
                        [0.0]
                    )
                ),

            "time_sec":
                round(
                    elapsed,
                    1
                ),
        })

    # -------------------------------------------------------
    # Paper-style CNN
    # -------------------------------------------------------

    if train_paper:

        history, elapsed = train_model(

            model_name=(
                "paper_cnn"
                if not args.smoke_test
                else "smoke_paper"
            ),

            model_builder=paper_builder,

            model_path=paper_path,

            epochs=epochs,

            batch_size=batch_size,

            augment=args.augment,

            subset_size=subset_size,
        )

        summaries.append({

            "model":
                "Paper-Style 1D CNN",

            "saved_to":
                str(paper_path),

            "best_val_acc":
                max(
                    history.history.get(
                        "val_accuracy",
                        [0.0]
                    )
                ),

            "time_sec":
                round(
                    elapsed,
                    1
                ),
        })

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

    print(
        f"\n{'=' * 60}"
    )

    print(
        "  TRAINING SUMMARY"
    )

    print(
        f"{'=' * 60}"
    )

    for s in summaries:

        print(
            f"\n  {s['model']}"
        )

        print(
            f"    Best Val Accuracy : "
            f"{s['best_val_acc']:.4f}"
        )

        print(
            f"    Training time     : "
            f"{s['time_sec']} s "
            f"({s['time_sec'] / 60:.1f} min)"
        )

        print(
            f"    Model saved to    : "
            f"{s['saved_to']}"
        )

    print(
        "\n  Run evaluation with:"
    )

    print(
        "    python evaluate.py"
    )


if __name__ == "__main__":
    main()
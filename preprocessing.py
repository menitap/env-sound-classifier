"""
preprocessing.py

UrbanSound8K dataset metadata processing, audio loading, and preprocessing module.

Responsible for:
- Loading and validating metadata from UrbanSound8K CSV.
- Constructing audio file paths across folds.
- Validating presence of audio files and fold directories.
- Partitioning metadata into reproducible training, validation, and test splits.
- Loading WAV audio, converting to mono, and resampling to target sample rate.
- Preprocessing audio with per-sample RMS normalization, padding,
  truncating, and reshaping to (88200, 1).
- Batch generation and dataset preparation for training, validation, and testing.
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf


REQUIRED_COLUMNS: List[str] = [
    "slice_file_name",
    "fold",
    "classID",
    "class",
]

TARGET_SAMPLE_RATE: int = 22050
TARGET_DURATION: int = 4
TARGET_SAMPLES: int = TARGET_SAMPLE_RATE * TARGET_DURATION  # 88200


def load_metadata(
    metadata_path: Union[str, Path],
) -> pd.DataFrame:
    """Load and validate the UrbanSound8K metadata CSV file."""

    path = Path(metadata_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read CSV at {path}: {exc}"
        ) from exc

    # Check required columns
    missing_cols = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Metadata is missing required column(s): {missing_cols}. "
            f"Present columns: {list(df.columns)}"
        )

    # Check missing values
    nan_counts = df[REQUIRED_COLUMNS].isna().sum()

    if nan_counts.any():
        bad_cols = nan_counts[nan_counts > 0].to_dict()

        raise ValueError(
            f"Metadata contains missing/NaN values: {bad_cols}"
        )

    # Validate folds
    if not pd.api.types.is_integer_dtype(df["fold"]):
        try:
            df["fold"] = df["fold"].astype(int)
        except Exception as exc:
            raise ValueError(
                "Column 'fold' must contain integer values."
            ) from exc

    invalid_folds = df[
        ~df["fold"].between(1, 10)
    ]["fold"].unique()

    if len(invalid_folds) > 0:
        raise ValueError(
            "Metadata contains invalid fold values "
            f"(must be 1-10): {invalid_folds.tolist()}"
        )

    # Validate class IDs
    if not pd.api.types.is_integer_dtype(df["classID"]):
        try:
            df["classID"] = df["classID"].astype(int)
        except Exception as exc:
            raise ValueError(
                "Column 'classID' must contain integer values."
            ) from exc

    invalid_classes = df[
        ~df["classID"].between(0, 9)
    ]["classID"].unique()

    if len(invalid_classes) > 0:
        raise ValueError(
            "Metadata contains invalid classID values "
            f"(must be 0-9): {invalid_classes.tolist()}"
        )

    return df


def get_audio_path(
    audio_root: Union[str, Path],
    fold: Union[int, str],
    filename: str,
) -> Path:
    """Construct the full filesystem path for an UrbanSound8K audio file."""

    fold_str = str(fold).strip()

    if not fold_str.lower().startswith("fold"):
        fold_dir = f"fold{fold_str}"
    else:
        fold_dir = fold_str

    return (
        Path(audio_root)
        / fold_dir
        / str(filename).strip()
    )


def validate_dataset(
    metadata: pd.DataFrame,
    audio_root: Union[str, Path],
) -> Dict[str, Union[int, List[str], Dict[str, int], bool]]:
    """Validate dataset folds and referenced audio files."""

    root = Path(audio_root)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Audio root directory does not exist: {root}"
        )

    # Check all 10 folds
    missing_folds: List[str] = []

    for f in range(1, 11):
        fold_path = root / f"fold{f}"

        if not fold_path.is_dir():
            missing_folds.append(f"fold{f}")

    # Check referenced audio files
    missing_files: List[str] = []

    for _, row in metadata.iterrows():
        expected_path = get_audio_path(
            root,
            row["fold"],
            row["slice_file_name"],
        )

        if not expected_path.is_file():
            missing_files.append(str(expected_path))

    class_counts = (
        metadata["class"]
        .value_counts()
        .to_dict()
    )

    fold_counts = (
        metadata["fold"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "total_recordings": len(metadata),
        "all_folds_exist": len(missing_folds) == 0,
        "missing_folds": missing_folds,
        "missing_files": missing_files,
        "class_distribution": class_counts,
        "fold_distribution": fold_counts,
    }


def create_data_splits(
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train, validation, and test splits."""

    if "fold" not in metadata.columns:
        raise ValueError(
            "Metadata DataFrame must contain a 'fold' column."
        )

    # UrbanSound8K split used by this project:
    # train = folds 1-8
    # validation = fold 9
    # test = fold 10

    train_folds = list(range(1, 9))
    val_folds = [9]
    test_folds = [10]

    train_df = (
        metadata[
            metadata["fold"].isin(train_folds)
        ]
        .copy()
        .reset_index(drop=True)
    )

    val_df = (
        metadata[
            metadata["fold"].isin(val_folds)
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        metadata[
            metadata["fold"].isin(test_folds)
        ]
        .copy()
        .reset_index(drop=True)
    )

    return train_df, val_df, test_df


def _resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio using soxr or scipy."""

    if orig_sr == target_sr:
        return audio.astype(np.float32)

    try:
        import soxr

        return soxr.resample(
            audio,
            orig_sr,
            target_sr,
        ).astype(np.float32)

    except Exception:
        import scipy.signal

        num_target = int(
            round(
                len(audio)
                * float(target_sr)
                / orig_sr
            )
        )

        return scipy.signal.resample(
            audio,
            num_target,
        ).astype(np.float32)


def load_audio(
    audio_path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """Load WAV audio, convert to mono, and resample."""

    if target_sr <= 0:
        raise ValueError(
            f"target_sr must be a positive integer, got {target_sr}"
        )

    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Audio file does not exist: {path}"
        )

    try:
        data, sr = sf.read(
            str(path),
            dtype="float32",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read/decode audio file {path}: {exc}"
        ) from exc

    # Convert stereo/multichannel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample
    if sr != target_sr:
        data = _resample(
            data,
            orig_sr=sr,
            target_sr=target_sr,
        )

    return data.astype(np.float32), target_sr


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sr: int = TARGET_SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    """
    Preprocess audio into fixed length (88200, 1).

    Processing:
    1. Validate input.
    2. Resample if necessary.
    3. Per-sample RMS normalization.
    4. Pad/truncate to exactly 4 seconds.
    5. Return float32 tensor of shape (88200, 1).
    """

    if sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be a positive integer, got {sample_rate}"
        )

    if target_sr <= 0:
        raise ValueError(
            f"target_sr must be a positive integer, got {target_sr}"
        )

    if target_samples <= 0:
        raise ValueError(
            f"target_samples must be a positive integer, got {target_samples}"
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    # Reject empty arrays
    if audio.size == 0:
        raise ValueError(
            "Audio array cannot be empty."
        )

    # Must be mono
    if audio.ndim != 1:
        raise ValueError(
            "preprocess_audio expects a 1D mono waveform, "
            f"but received array with shape {audio.shape}. "
            "Convert multichannel audio to mono before preprocessing."
        )

    # Resample if needed
    if sample_rate != target_sr:
        audio = _resample(
            audio,
            orig_sr=sample_rate,
            target_sr=target_sr,
        )

    # ============================================================
    # PER-SAMPLE RMS NORMALIZATION
    # ============================================================
    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    if rms > 1e-6:
        audio = audio / rms

    # Prevent extreme values
    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    # ============================================================
    # FIXED LENGTH: PAD OR TRUNCATE
    # ============================================================
    n_samples = len(audio)

    if n_samples < target_samples:
        audio = np.pad(
            audio,
            (
                0,
                target_samples - n_samples,
            ),
            mode="constant",
        )

    elif n_samples > target_samples:
        audio = audio[:target_samples]

    # Final shape
    return audio.astype(
        np.float32
    ).reshape(
        target_samples,
        1,
    )


def load_and_preprocess_audio(
    audio_path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    """Load and preprocess one audio file."""

    audio, sr = load_audio(
        audio_path,
        target_sr=target_sr,
    )

    return preprocess_audio(
        audio,
        sample_rate=sr,
        target_sr=target_sr,
        target_samples=target_samples,
    )


def load_and_preprocess(audio_path):
    """Compatibility wrapper for the training pipeline."""

    return load_and_preprocess_audio(audio_path)


def create_batch_generator(
    metadata: pd.DataFrame,
    audio_root: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield batches of preprocessed audio and class labels."""

    if not isinstance(
        batch_size,
        (int, np.integer),
    ) or batch_size <= 0:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size}"
        )

    for col in [
        "slice_file_name",
        "fold",
        "classID",
    ]:
        if col not in metadata.columns:
            raise ValueError(
                f"Metadata is missing required column: '{col}'"
            )

    n_samples = len(metadata)

    if n_samples == 0:
        return

    # Determine index ordering
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    # Process batches
    for start_idx in range(
        0,
        n_samples,
        batch_size,
    ):
        end_idx = min(
            start_idx + batch_size,
            n_samples,
        )

        batch_indices = indices[
            start_idx:end_idx
        ]

        current_batch_size = len(
            batch_indices
        )

        batch_x = np.empty(
            (
                current_batch_size,
                TARGET_SAMPLES,
                1,
            ),
            dtype=np.float32,
        )

        batch_y = np.empty(
            current_batch_size,
            dtype=np.int64,
        )

        for i, idx in enumerate(batch_indices):
            row = metadata.iloc[idx]

            filename = row["slice_file_name"]
            fold = row["fold"]
            class_id = int(row["classID"])

            if not (0 <= class_id <= 9):
                raise ValueError(
                    f"Invalid classID {class_id} "
                    f"for recording '{filename}'. "
                    "Must be in 0-9."
                )

            audio_path = get_audio_path(
                audio_root,
                fold,
                filename,
            )

            try:
                processed_waveform = (
                    load_and_preprocess_audio(
                        audio_path
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Error processing audio file "
                    f"'{filename}' in fold '{fold}' "
                    f"at path '{audio_path}': {exc}"
                ) from exc

            batch_x[i] = processed_waveform
            batch_y[i] = class_id

        yield batch_x, batch_y


def prepare_datasets(
    metadata_path: Union[str, Path],
    audio_root: Union[str, Path],
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Prepare UrbanSound8K train/validation/test datasets."""

    if not isinstance(
        batch_size,
        (int, np.integer),
    ) or batch_size <= 0:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size}"
        )

    metadata = load_metadata(
        metadata_path
    )

    val_results = validate_dataset(
        metadata,
        audio_root,
    )

    # Dataset integrity checks
    if not val_results["all_folds_exist"]:
        missing_folds = (
            val_results["missing_folds"]
        )

        raise FileNotFoundError(
            "Dataset validation failed: "
            "missing fold directory/directories: "
            f"{missing_folds}"
        )

    if val_results["missing_files"]:
        missing_count = len(
            val_results["missing_files"]
        )

        sample_missing = (
            val_results["missing_files"][:5]
        )

        raise FileNotFoundError(
            "Dataset validation failed: "
            f"missing {missing_count} audio file(s). "
            f"Sample missing files: {sample_missing}"
        )

    # Create splits
    train_df, val_df, test_df = (
        create_data_splits(metadata)
    )

    split_sizes = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "total": len(metadata),
    }

    # Fresh train generator
    def get_train_generator(
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        return create_batch_generator(
            train_df,
            audio_root=audio_root,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    # Deterministic validation generator
    def get_val_generator():
        return create_batch_generator(
            val_df,
            audio_root=audio_root,
            batch_size=batch_size,
            shuffle=False,
            seed=None,
        )

    # Deterministic test generator
    def get_test_generator():
        return create_batch_generator(
            test_df,
            audio_root=audio_root,
            batch_size=batch_size,
            shuffle=False,
            seed=None,
        )

    return {
        "metadata": metadata,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_generator_fn": get_train_generator,
        "val_generator_fn": get_val_generator,
        "test_generator_fn": get_test_generator,
        "validation_results": val_results,
        "split_sizes": split_sizes,
        "num_classes": len(
            metadata["classID"].unique()
        ),
        "target_shape": (
            TARGET_SAMPLES,
            1,
        ),
        "batch_size": batch_size,
    }
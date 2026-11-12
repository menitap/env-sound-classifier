"""preprocessing.py

UrbanSound8K dataset metadata processing, audio loading, and preprocessing module.
Responsible for:
- Loading and validating metadata from UrbanSound8K CSV.
- Constructing audio file paths across folds.
- Validating presence of audio files and fold directories.
- Partitioning metadata into reproducible training, validation, and test splits.
- Loading WAV audio, converting to mono, and resampling to target sample rate.
- Preprocessing audio (padding, truncating, reshaping to (88200, 1), float32).
- Batch generation and dataset preparation for training, validation, and testing.
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import soundfile as sf

REQUIRED_COLUMNS: List[str] = ["slice_file_name", "fold", "classID", "class"]

TARGET_SAMPLE_RATE: int = 22050
TARGET_DURATION: int = 4
TARGET_SAMPLES: int = TARGET_SAMPLE_RATE * TARGET_DURATION  # 88200


def load_metadata(metadata_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate the UrbanSound8K metadata CSV file.

    Parameters
    ----------
    metadata_path : str or Path
        Path to the UrbanSound8K.csv metadata file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing verified metadata.

    Raises
    ------
    FileNotFoundError
        If metadata_path does not exist or is not a file.
    ValueError
        If required columns are missing, missing/NaN values are detected,
        fold values are not within 1-10, or classID values are not integers in 0-9.
    """
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at {path}: {exc}") from exc

    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Metadata is missing required column(s): {missing_cols}. "
            f"Present columns: {list(df.columns)}"
        )

    # Check for missing values in required columns
    nan_counts = df[REQUIRED_COLUMNS].isna().sum()
    if nan_counts.any():
        bad_cols = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(f"Metadata contains missing/NaN values: {bad_cols}")

    # Validate fold values: must be integers within 1-10
    if not pd.api.types.is_integer_dtype(df["fold"]):
        try:
            df["fold"] = df["fold"].astype(int)
        except Exception as exc:
            raise ValueError("Column 'fold' must contain integer values.") from exc

    invalid_folds = df[~df["fold"].between(1, 10)]["fold"].unique()
    if len(invalid_folds) > 0:
        raise ValueError(
            f"Metadata contains invalid fold values (must be 1-10): {invalid_folds.tolist()}"
        )

    # Validate classID values: must be integers within 0-9
    if not pd.api.types.is_integer_dtype(df["classID"]):
        try:
            df["classID"] = df["classID"].astype(int)
        except Exception as exc:
            raise ValueError("Column 'classID' must contain integer values.") from exc

    invalid_classes = df[~df["classID"].between(0, 9)]["classID"].unique()
    if len(invalid_classes) > 0:
        raise ValueError(
            f"Metadata contains invalid classID values (must be 0-9): {invalid_classes.tolist()}"
        )

    return df


def get_audio_path(audio_root: Union[str, Path], fold: Union[int, str], filename: str) -> Path:
    """Construct the full filesystem path for an UrbanSound8K audio file.

    Parameters
    ----------
    audio_root : str or Path
        Root directory containing the fold subdirectories (e.g. '.../audio').
    fold : int or str
        Fold identifier (e.g., 1 or 'fold1').
    filename : str
        Audio slice filename (e.g. '100032-3-0-0.wav').

    Returns
    -------
    Path
        Absolute or resolved Path to the audio file.
    """
    fold_str = str(fold).strip()
    if not fold_str.lower().startswith("fold"):
        fold_dir = f"fold{fold_str}"
    else:
        fold_dir = fold_str

    return Path(audio_root) / fold_dir / str(filename).strip()


def validate_dataset(
    metadata: pd.DataFrame, audio_root: Union[str, Path]
) -> Dict[str, Union[int, List[str], Dict[str, int], bool]]:
    """Validate that fold directories and referenced audio files exist on disk.

    Parameters
    ----------
    metadata : pd.DataFrame
        UrbanSound8K metadata DataFrame.
    audio_root : str or Path
        Root directory containing the 10 fold folders.

    Returns
    -------
    dict
        Validation summary containing:
        - 'total_recordings': int
        - 'all_folds_exist': bool
        - 'missing_folds': list of missing fold directory names
        - 'missing_files': list of missing file paths (as strings)
        - 'class_distribution': dict mapping class name to count
        - 'fold_distribution': dict mapping fold to count

    Raises
    ------
    FileNotFoundError
        If audio_root does not exist.
    """
    root = Path(audio_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Audio root directory does not exist: {root}")

    # Check that all 10 fold directories exist
    missing_folds: List[str] = []
    for f in range(1, 11):
        fold_path = root / f"fold{f}"
        if not fold_path.is_dir():
            missing_folds.append(f"fold{f}")

    # Check whether each audio file referenced in metadata exists
    missing_files: List[str] = []
    for _, row in metadata.iterrows():
        expected_path = get_audio_path(root, row["fold"], row["slice_file_name"])
        if not expected_path.is_file():
            missing_files.append(str(expected_path))

    class_counts = metadata["class"].value_counts().to_dict()
    fold_counts = metadata["fold"].value_counts().sort_index().to_dict()

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
    """Partition metadata into training, validation, and test splits.

    UrbanSound8K standard fold assignments are preserved:
    - Training: folds 1-8
    - Validation: fold 9
    - Testing: fold 10

    Parameters
    ----------
    metadata : pd.DataFrame
        UrbanSound8K metadata containing a 'fold' column.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (train_df, val_df, test_df) with reset indices.

    Raises
    ------
    ValueError
        If 'fold' column is missing or required folds are not found.
    """
    if "fold" not in metadata.columns:
        raise ValueError("Metadata DataFrame must contain a 'fold' column.")

    train_folds = list(range(1, 9))
    val_folds = [9]
    test_folds = [10]

    train_df = metadata[metadata["fold"].isin(train_folds)].copy().reset_index(drop=True)
    val_df = metadata[metadata["fold"].isin(val_folds)].copy().reset_index(drop=True)
    test_df = metadata[metadata["fold"].isin(test_folds)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Internal helper to resample audio using soxr or scipy.

    Parameters
    ----------
    audio : np.ndarray
        1D float32 audio waveform.
    orig_sr : int
        Original sample rate.
    target_sr : int
        Target sample rate.

    Returns
    -------
    np.ndarray
        Resampled 1D float32 waveform.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    try:
        import soxr

        return soxr.resample(audio, orig_sr, target_sr).astype(np.float32)
    except Exception:
        import scipy.signal

        num_target = int(round(len(audio) * float(target_sr) / orig_sr))
        return scipy.signal.resample(audio, num_target).astype(np.float32)


def load_audio(
    audio_path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """Load a WAV audio file, convert stereo to mono, and resample to target_sr.

    Parameters
    ----------
    audio_path : str or Path
        Path to the WAV audio file.
    target_sr : int, default=22050
        Desired output sampling rate.

    Returns
    -------
    waveform : np.ndarray
        1D audio array of shape (N,) and dtype float32.
    sample_rate : int
        Output sample rate (target_sr).

    Raises
    ------
    ValueError
        If target_sr is <= 0.
    FileNotFoundError
        If audio_path does not exist.
    RuntimeError
        If audio file cannot be decoded or is corrupted.
    """
    if target_sr <= 0:
        raise ValueError(f"target_sr must be a positive integer, got {target_sr}")

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception as exc:
        raise RuntimeError(f"Failed to read/decode audio file {path}: {exc}") from exc

    # Convert stereo/multichannel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to target_sr if required
    if sr != target_sr:
        data = _resample(data, orig_sr=sr, target_sr=target_sr)

    return data.astype(np.float32), target_sr


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sr: int = TARGET_SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    """Preprocess audio waveform into fixed length (88200, 1) float32 array.

    Enforces target sample rate, pads shorter recordings with zeros,
    truncates longer recordings, and reshapes to (target_samples, 1).

    Parameters
    ----------
    audio : np.ndarray
        Input 1D mono audio waveform.
    sample_rate : int
        Sampling rate of the input audio. Must be > 0.
    target_sr : int, default=22050
        Expected sampling rate. Must be > 0. If sample_rate != target_sr, audio is resampled.
    target_samples : int, default=88200
        Expected number of samples (4 seconds at 22050 Hz). Must be > 0.

    Returns
    -------
    np.ndarray
        Processed audio array of shape (target_samples, 1) and dtype float32.

    Raises
    ------
    ValueError
        If sample_rate, target_sr, or target_samples are <= 0,
        if audio is empty (size 0), or if audio is not 1D (multichannel).
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate}")
    if target_sr <= 0:
        raise ValueError(f"target_sr must be a positive integer, got {target_sr}")
    if target_samples <= 0:
        raise ValueError(f"target_samples must be a positive integer, got {target_samples}")

    audio = np.asarray(audio, dtype=np.float32)

    # Reject empty arrays
    if audio.size == 0:
        raise ValueError("Audio array cannot be empty.")

    # Reject multichannel audio: preprocess_audio expects mono input
    if audio.ndim != 1:
        raise ValueError(
            f"preprocess_audio expects a 1D mono waveform, but received array with shape {audio.shape}. "
            "Convert multichannel audio to mono before preprocessing."
        )

    # Resample if sample rate does not match target
    if sample_rate != target_sr:
        audio = _resample(audio, orig_sr=sample_rate, target_sr=target_sr)

    # Enforce exactly target_samples: pad if short, truncate if long
    n_samples = len(audio)
    if n_samples < target_samples:
        audio = np.pad(audio, (0, target_samples - n_samples), mode="constant")
    elif n_samples > target_samples:
        audio = audio[:target_samples]

    # Convert to float32 and reshape to (target_samples, 1)
    return audio.astype(np.float32).reshape(target_samples, 1)


def load_and_preprocess_audio(
    audio_path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    """Combine audio loading and preprocessing into a single pipeline step.

    Parameters
    ----------
    audio_path : str or Path
        Path to the WAV audio file.
    target_sr : int, default=22050
        Desired output sampling rate (22050 Hz). Must be > 0.
    target_samples : int, default=88200
        Target sample count (88200 samples = 4 seconds). Must be > 0.

    Returns
    -------
    np.ndarray
        Processed waveform of shape (88200, 1) and dtype float32.
    """
    audio, sr = load_audio(audio_path, target_sr=target_sr)
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
    """Yield batches of preprocessed audio recordings and integer class labels on demand.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame containing UrbanSound8K slice metadata ('slice_file_name', 'fold', 'classID').
    audio_root : str or Path
        Root directory containing the fold subfolders.
    batch_size : int, default=32
        Number of audio recordings per batch. Must be > 0.
    shuffle : bool, default=False
        Whether to shuffle the recordings prior to batch generation.
    seed : int, optional
        Random seed for reproducible shuffling.

    Yields
    ------
    batch_x : np.ndarray
        Audio tensor of shape (current_batch_size, 88200, 1) with dtype float32.
    batch_y : np.ndarray
        Integer class label array of shape (current_batch_size,) with dtype int64.

    Raises
    ------
    ValueError
        If batch_size is not a positive integer, required columns are missing,
        or classID values are outside 0-9.
    FileNotFoundError
        If an audio file in the batch does not exist.
    RuntimeError
        If an audio file cannot be loaded or is corrupted.
    """
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

    for col in ["slice_file_name", "fold", "classID"]:
        if col not in metadata.columns:
            raise ValueError(f"Metadata is missing required column: '{col}'")

    n_samples = len(metadata)
    if n_samples == 0:
        return

    # Determine index ordering
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    # Process in batches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        current_batch_size = len(batch_indices)

        batch_x = np.empty((current_batch_size, TARGET_SAMPLES, 1), dtype=np.float32)
        batch_y = np.empty(current_batch_size, dtype=np.int64)

        for i, idx in enumerate(batch_indices):
            row = metadata.iloc[idx]
            filename = row["slice_file_name"]
            fold = row["fold"]
            class_id = int(row["classID"])

            if not (0 <= class_id <= 9):
                raise ValueError(
                    f"Invalid classID {class_id} for recording '{filename}'. Must be in 0-9."
                )

            audio_path = get_audio_path(audio_root, fold, filename)
            try:
                processed_waveform = load_and_preprocess_audio(audio_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Error processing audio file '{filename}' in fold '{fold}' "
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
    """Load metadata, validate files, partition into train/val/test, and provide batch generators.

    UrbanSound8K standard fold assignments:
    - Training: folds 1-8
    - Validation: fold 9
    - Testing: fold 10

    Parameters
    ----------
    metadata_path : str or Path
        Path to the UrbanSound8K.csv metadata file.
    audio_root : str or Path
        Root directory containing fold subdirectories (fold1 to fold10).
    batch_size : int, default=32
        Batch size for generators. Must be a positive integer.

    Returns
    -------
    dict
        Dataset pipeline dictionary containing:
        - 'metadata': complete pd.DataFrame
        - 'train_df': pd.DataFrame for training
        - 'val_df': pd.DataFrame for validation
        - 'test_df': pd.DataFrame for testing
        - 'train_generator_fn': Callable[..., Generator] producing fresh train generators
        - 'val_generator_fn': Callable[..., Generator] producing fresh validation generators
        - 'test_generator_fn': Callable[..., Generator] producing fresh test generators
        - 'validation_results': dict from validate_dataset()
        - 'split_sizes': dict with 'train', 'val', 'test', 'total' counts
        - 'num_classes': int (10)
        - 'target_shape': tuple (88200, 1)
        - 'batch_size': int

    Raises
    ------
    FileNotFoundError
        If any fold directories or audio files are missing on disk.
    """
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

    metadata = load_metadata(metadata_path)
    val_results = validate_dataset(metadata, audio_root)

    # Enforce dataset integrity: do not silently continue if dataset is incomplete
    if not val_results["all_folds_exist"]:
        missing_folds = val_results["missing_folds"]
        raise FileNotFoundError(
            f"Dataset validation failed: missing fold directory/directories: {missing_folds}"
        )

    if val_results["missing_files"]:
        missing_count = len(val_results["missing_files"])
        sample_missing = val_results["missing_files"][:5]
        raise FileNotFoundError(
            f"Dataset validation failed: missing {missing_count} audio file(s). "
            f"Sample missing files: {sample_missing}"
        )

    train_df, val_df, test_df = create_data_splits(metadata)

    split_sizes = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "total": len(metadata),
    }

    # Factory functions for fresh generators across epochs
    def get_train_generator(shuffle: bool = True, seed: Optional[int] = None):
        return create_batch_generator(
            train_df,
            audio_root=audio_root,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    def get_val_generator():
        # Validation ordering must be deterministic
        return create_batch_generator(
            val_df,
            audio_root=audio_root,
            batch_size=batch_size,
            shuffle=False,
            seed=None,
        )

    def get_test_generator():
        # Test ordering must be deterministic
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
        "num_classes": len(metadata["classID"].unique()),
        "target_shape": (TARGET_SAMPLES, 1),
        "batch_size": batch_size,
    }

import os
import numpy as np
import pandas as pd

from preprocessing import load_and_preprocess
from training_augmentation import augment_audio


# Project and dataset paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(PROJECT_DIR), "archive")

METADATA_PATH = os.path.join(DATASET_PATH, "UrbanSound8K.csv")


def load_dataset(folds, augmented=False):
    """
    Load UrbanSound8K data for the specified folds.

    Parameters
    ----------
    folds : list
        List of fold numbers to load.
        Example: [1, 2, 3, 4, 5, 6, 7, 8]

    augmented : bool
        If True, apply training-time augmentation.
        This should only be True for training data.

    Returns
    -------
    X : numpy.ndarray
        Audio data with shape:
        (number_of_samples, 88200, 1)

    y : numpy.ndarray
        Integer class labels from 0 to 9.
    """

    metadata = pd.read_csv(METADATA_PATH)

    X = []
    y = []

    selected = metadata[metadata["fold"].isin(folds)]

    print(f"Loading folds: {folds}")
    print(f"Number of audio files: {len(selected)}")

    for _, row in selected.iterrows():

        audio_path = os.path.join(
            DATASET_PATH,
            f"fold{row['fold']}",
            row["slice_file_name"]
        )

        try:
            # Load and preprocess audio
            audio = load_and_preprocess(audio_path)

            # Apply augmentation only when requested
            if augmented:
                audio = augment_audio(audio)

            X.append(audio)
            y.append(int(row["classID"]))

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print("Loaded X shape:", X.shape)
    print("Loaded y shape:", y.shape)

    return X, y


def load_train_val_test(augmented=False):
    """
    Load the complete train, validation, and test datasets.

    Dataset split:
        Training   -> folds 1-8
        Validation -> fold 9
        Testing    -> fold 10

    Parameters
    ----------
    augmented : bool
        If True, apply training augmentation to the training set.
        Validation and test sets are always kept unaugmented.

    Returns
    -------
    X_train, y_train
    X_val, y_val
    X_test, y_test
    """

    print("\n========== TRAINING DATA ==========")

    X_train, y_train = load_dataset(
        folds=[1, 2, 3, 4, 5, 6, 7, 8],
        augmented=augmented
    )

    print("\n========== VALIDATION DATA ==========")

    X_val, y_val = load_dataset(
        folds=[9],
        augmented=False
    )

    print("\n========== TEST DATA ==========")

    X_test, y_test = load_dataset(
        folds=[10],
        augmented=False
    )

    print("\n========== DATASET SUMMARY ==========")

    print("Training:")
    print("  X_train:", X_train.shape)
    print("  y_train:", y_train.shape)

    print("Validation:")
    print("  X_val:", X_val.shape)
    print("  y_val:", y_val.shape)

    print("Testing:")
    print("  X_test:", X_test.shape)
    print("  y_test:", y_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )
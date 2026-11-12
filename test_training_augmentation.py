import os
import pandas as pd
import numpy as np

from preprocessing import load_and_preprocess
from training_augmentation import augment_audio

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(PROJECT_DIR), "archive")

metadata = pd.read_csv(
    os.path.join(DATASET_PATH, "UrbanSound8K.csv")
)

row = metadata.iloc[0]

audio_path = os.path.join(
    DATASET_PATH,
    f"fold{row['fold']}",
    row["slice_file_name"]
)

# Preprocess
audio = load_and_preprocess(audio_path)

# Augment
augmented_audio = augment_audio(audio)

print("Original shape:", audio.shape)
print("Augmented shape:", augmented_audio.shape)
print("Original dtype:", audio.dtype)
print("Augmented dtype:", augmented_audio.dtype)

print("\nOriginal range:",
      np.min(audio), "to", np.max(audio))

print("Augmented range:",
      np.min(augmented_audio), "to", np.max(augmented_audio))

print("\nTraining augmentation test successful!")
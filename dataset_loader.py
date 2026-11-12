import os
import pandas as pd
import librosa

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(PROJECT_DIR), "archive")

metadata_path = os.path.join(DATASET_PATH, "UrbanSound8K.csv")
metadata = pd.read_csv(metadata_path)

print("Dataset loaded successfully!")
print("Number of audio files:", len(metadata))
print("\nClasses:")
print(metadata["class"].value_counts())

# Test loading one audio file
row = metadata.iloc[0]

audio_path = os.path.join(
    DATASET_PATH,
    f"fold{row['fold']}",
    row["slice_file_name"]
)

audio, sample_rate = librosa.load(
    audio_path,
    sr=22050,
    mono=True
)

print("\nTest audio:")
print("File:", row["slice_file_name"])
print("Class:", row["class"])
print("Audio shape:", audio.shape)
print("Sample rate:", sample_rate)
print("Duration:", len(audio) / sample_rate, "seconds")
import os
import pandas as pd

from preprocessing import load_and_preprocess

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

audio = load_and_preprocess(audio_path)

print("File:", row["slice_file_name"])
print("Class:", row["class"])
print("Processed shape:", audio.shape)
print("Expected shape:", (88200, 1))
print("Data type:", audio.dtype)
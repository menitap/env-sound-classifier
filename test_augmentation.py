import os
import pandas as pd
import librosa
import soundfile as sf

from augmentation import add_noise, change_volume, time_shift

# Dataset location
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(PROJECT_DIR), "archive")

# Load metadata
metadata = pd.read_csv(
    os.path.join(DATASET_PATH, "UrbanSound8K.csv")
)

# Pick one real audio file
row = metadata.iloc[0]

audio_path = os.path.join(
    DATASET_PATH,
    f"fold{row['fold']}",
    row["slice_file_name"]
)

# Load audio
audio, sr = librosa.load(
    audio_path,
    sr=22050,
    mono=True
)

print("Original audio:")
print("File:", row["slice_file_name"])
print("Class:", row["class"])
print("Samples:", len(audio))
print("Sample rate:", sr)
print("Duration:", len(audio) / sr)

# -----------------------------
# Apply augmentation
# -----------------------------

noisy_audio = add_noise(audio, snr_db=20)
volume_audio = change_volume(audio, factor=1.2)
shifted_audio = time_shift(audio, shift_max=0.2)

# Save augmented examples
output_dir = os.path.join(PROJECT_DIR, "augmentation_test")
os.makedirs(output_dir, exist_ok=True)

sf.write(
    os.path.join(output_dir, "original.wav"),
    audio,
    sr
)

sf.write(
    os.path.join(output_dir, "noisy_20db.wav"),
    noisy_audio,
    sr
)

sf.write(
    os.path.join(output_dir, "volume_changed.wav"),
    volume_audio,
    sr
)

sf.write(
    os.path.join(output_dir, "time_shifted.wav"),
    shifted_audio,
    sr
)

print("\nAugmentation successful!")
print("Saved files to:", output_dir)
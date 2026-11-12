import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import load_and_preprocess
from augmentation import add_noise, change_volume, time_shift


# Project and dataset paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(PROJECT_DIR), "archive")


# Load UrbanSound8K metadata
metadata = pd.read_csv(
    os.path.join(DATASET_PATH, "UrbanSound8K.csv")
)

# Select one example audio file
row = metadata.iloc[0]

audio_path = os.path.join(
    DATASET_PATH,
    f"fold{row['fold']}",
    row["slice_file_name"]
)


# Load and preprocess audio
audio = load_and_preprocess(audio_path).squeeze()

# Apply augmentations
noisy_audio = add_noise(audio, snr_db=20)
volume_audio = change_volume(audio, factor=1.2)
shifted_audio = time_shift(audio, shift_max=0.2)


# Time axis in seconds
sample_rate = 22050
time = np.arange(len(audio)) / sample_rate


# Create visualization
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(time, audio)
plt.title(f"Original Audio - {row['class']}")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time, noisy_audio)
plt.title("Gaussian Noise Added - 20 dB SNR")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(time, volume_audio)
plt.title("Volume Variation")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time, shifted_audio)
plt.title("Time Shift")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()


# Save visualization
output_path = os.path.join(
    PROJECT_DIR,
    "augmentation_waveforms.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


print("\nWaveform visualization created successfully!")
print("Saved to:", output_path)
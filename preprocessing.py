import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION = 4
NUM_SAMPLES = SAMPLE_RATE * DURATION


def load_and_preprocess(audio_path):
    """
    Load an audio file and convert it to the standard
    4-second, 22050 Hz waveform expected by the models.
    """

    # Load audio and resample
    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Make sure audio is exactly 4 seconds
    if len(audio) < NUM_SAMPLES:
        # Zero-pad shorter clips
        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )
    else:
        # Truncate longer clips
        audio = audio[:NUM_SAMPLES]

    # Convert to float32
    audio = audio.astype(np.float32)

    # Add channel dimension for Conv1D
    audio = audio.reshape(NUM_SAMPLES, 1)

    return audio
import numpy as np


def add_noise(audio, snr_db=20):
    """
    Add Gaussian noise to an audio signal at a specified SNR.
    """

    audio = np.asarray(audio, dtype=np.float32)

    signal_power = np.mean(audio ** 2)

    if signal_power == 0:
        return audio

    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        size=audio.shape
    ).astype(np.float32)

    noisy_audio = audio + noise

    # Prevent clipping
    max_val = np.max(np.abs(noisy_audio))

    if max_val > 1:
        noisy_audio = noisy_audio / max_val

    return noisy_audio.astype(np.float32)


def change_volume(audio, factor=1.2):
    """
    Change the volume of an audio signal.
    """

    audio = np.asarray(audio, dtype=np.float32)

    augmented = audio * factor

    return np.clip(
        augmented,
        -1.0,
        1.0
    ).astype(np.float32)


def time_shift(audio, shift_max=0.2):
    """
    Randomly shift the audio signal left or right.
    """

    audio = np.asarray(audio, dtype=np.float32)

    shift = np.random.randint(
        -int(len(audio) * shift_max),
        int(len(audio) * shift_max)
    )

    return np.roll(audio, shift).astype(np.float32)
import numpy as np

from augmentation import add_noise, change_volume, time_shift


def augment_audio(
    audio,
    use_noise=True,
    use_volume=True,
    use_time_shift=True
):
    """
    Apply training-time augmentation to a preprocessed audio waveform.
    """

    # Remove channel dimension temporarily
    audio = audio.squeeze()

    if use_noise:
        audio = add_noise(audio, snr_db=20)

    if use_volume:
        audio = change_volume(audio, factor=1.2)

    if use_time_shift:
        audio = time_shift(audio, shift_max=0.2)

    # Restore channel dimension
    audio = audio.reshape(-1, 1)

    return audio.astype(np.float32)
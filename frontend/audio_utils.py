"""frontend.audio_utils

Audio processing bridge connecting Streamlit file uploaders with preprocessing.py.
Reuses existing preprocessing routines:
- load_audio()
- preprocess_audio()
- load_and_preprocess_audio()
"""

import io
from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import soundfile as sf

from frontend.config import TARGET_DURATION, TARGET_SAMPLE_RATE, TARGET_SAMPLES
from preprocessing import preprocess_audio


def read_uploaded_audio(
    uploaded_file,
) -> Tuple[np.ndarray, int, Dict[str, Union[float, int, str]]]:
    """Read an audio file from a Streamlit UploadedFile or local path.

    Converts stereo to mono (preserving preprocessing.py logic) without
    premature padding or truncation, and extracts native audio metadata.

    Parameters
    ----------
    uploaded_file : UploadedFile or str or Path
        Streamlit file buffer or file path.

    Returns
    -------
    raw_mono : np.ndarray
        1D float32 audio waveform at original sample rate.
    sample_rate : int
        Native sample rate of the file.
    metadata : dict
        File details: duration, original_sr, channels, num_samples, file_name, file_size_kb.
    """
    file_name = getattr(uploaded_file, "name", str(uploaded_file))
    file_size_kb = 0.0

    if hasattr(uploaded_file, "getvalue"):
        # Streamlit UploadedFile in-memory buffer
        bytes_data = uploaded_file.getvalue()
        file_size_kb = len(bytes_data) / 1024.0
        buffer = io.BytesIO(bytes_data)
        data, sr = sf.read(buffer, dtype="float32")
    else:
        # Filesystem path
        p = Path(uploaded_file)
        if p.is_file():
            file_size_kb = p.stat().st_size / 1024.0
        data, sr = sf.read(str(uploaded_file), dtype="float32")

    orig_channels = 1 if data.ndim == 1 else data.shape[1]
    orig_samples = len(data)
    duration_sec = orig_samples / sr if sr > 0 else 0.0

    # Convert to mono if multichannel (consistent with load_audio)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    raw_mono = data.astype(np.float32)

    metadata = {
        "file_name": file_name,
        "file_size_kb": file_size_kb,
        "duration_sec": duration_sec,
        "original_sr": sr,
        "channels": "Mono" if orig_channels == 1 else f"Stereo ({orig_channels}ch)",
        "original_samples": orig_samples,
    }

    return raw_mono, sr, metadata


def process_audio_for_inference(
    raw_audio: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process an audio waveform into the model input shape (1, 88200, 1).

    Reuses preprocessing.preprocess_audio().

    Parameters
    ----------
    raw_audio : np.ndarray
        1D mono float32 waveform.
    sample_rate : int
        Sampling rate of raw_audio.

    Returns
    -------
    input_tensor : np.ndarray
        Shape (1, 88200, 1) float32 array ready for model inference.
    processed_1d : np.ndarray
        Shape (88200,) 1D float32 array suitable for waveform plotting.
    """
    # preprocess_audio returns (88200, 1) float32
    processed = preprocess_audio(
        raw_audio,
        sample_rate=sample_rate,
        target_sr=TARGET_SAMPLE_RATE,
        target_samples=TARGET_SAMPLES,
    )

    # 1D slice for visualization
    processed_1d = processed.squeeze(axis=-1)

    # Batch tensor for model prediction: (1, 88200, 1)
    input_tensor = np.expand_dims(processed, axis=0)

    return input_tensor, processed_1d


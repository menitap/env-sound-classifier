"""
feature_extraction.py
Extracts 250-dimensional hand-crafted features from preprocessed audio waveforms.

Runs in .venv_ml (requires librosa). Never imports TensorFlow/Keras.
Input: preprocessed waveforms as numpy arrays (shape: [88200] or [88200,1], float32)
Output: 250-dim feature vector per clip.

Feature layout (must match ml_config.FEATURE_DIM = 250):
  [0:40]   MFCC means         (N_MFCC=40)
  [40:80]  MFCC delta means   (N_MFCC=40)
  [80:120] MFCC delta2 means  (N_MFCC=40)
  [120:132] Chroma STFT means (N_CHROMA=12)
  [132:139] Spectral contrast means (n_bands+1=7, fmin=CONTRAST_FMIN=100)
  [139:193] Mel spectrogram means  (N_MEL=54)
  [193:247] Mel spectrogram vars   (N_MEL=54)
  [247]    ZCR mean
  [248]    RMS mean
  [249]    RMS variance
  Total: 40+40+40+12+7+54+54+1+1+1 = 250
"""

import numpy as np

import librosa

from ml_config import (
    SAMPLE_RATE, N_MFCC, N_CHROMA, N_CONTRAST_BANDS,
    CONTRAST_FMIN, N_MEL, FEATURE_DIM
)


def extract_features(waveform: np.ndarray) -> np.ndarray:
    """
    Extract 250-dimensional feature vector from a single waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Preprocessed audio, shape (88200,) or (88200, 1), dtype float32.
        Must already be RMS-normalised and padded/truncated to 88200 samples.

    Returns
    -------
    features : np.ndarray  shape (250,), dtype float32
        Will contain only finite values if extraction succeeds normally.
    """
    # Flatten to 1-D float32
    y = np.asarray(waveform, dtype=np.float32).squeeze()
    sr = SAMPLE_RATE

    # ---- MFCC + deltas ----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)       # (40, T)
    mfcc_delta  = librosa.feature.delta(mfcc)                      # (40, T)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)             # (40, T)
    mfcc_mean   = mfcc.mean(axis=1)        # (40,)
    delta_mean  = mfcc_delta.mean(axis=1)  # (40,)
    delta2_mean = mfcc_delta2.mean(axis=1) # (40,)

    # ---- Chroma STFT ----
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)  # (12, T)
    chroma_mean = chroma.mean(axis=1)  # (12,)

    # ---- Spectral contrast ----
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_bands=N_CONTRAST_BANDS, fmin=CONTRAST_FMIN
    )  # (7, T)
    contrast_mean = contrast.mean(axis=1)  # (7,)

    # ---- Mel spectrogram ----
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)  # (54, T)
    mel_mean = mel.mean(axis=1)  # (54,)
    mel_var  = mel.var(axis=1)   # (54,)

    # ---- Zero Crossing Rate ----
    zcr = librosa.feature.zero_crossing_rate(y)  # (1, T)
    zcr_mean = np.array([zcr.mean()], dtype=np.float32)  # (1,)

    # ---- RMS Energy ----
    rms = librosa.feature.rms(y=y)  # (1, T)
    rms_mean = np.array([rms.mean()], dtype=np.float32)   # (1,)
    rms_var  = np.array([rms.var()], dtype=np.float32)    # (1,)

    # ---- Concatenate → 250-dim vector ----
    features = np.concatenate([
        mfcc_mean,     # 40
        delta_mean,    # 40
        delta2_mean,   # 40
        chroma_mean,   # 12
        contrast_mean, # 7
        mel_mean,      # 54
        mel_var,       # 54
        zcr_mean,      # 1
        rms_mean,      # 1
        rms_var,       # 1
    ]).astype(np.float32)

    assert features.shape == (FEATURE_DIM,), (
        f"Feature dim mismatch: expected {FEATURE_DIM}, got {features.shape[0]}"
    )
    return features

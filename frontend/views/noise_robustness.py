"""frontend.views.noise_robustness

View for Noise Robustness Testing, interactive SNR degradation, and clean vs noisy comparison.
"""

from typing import Tuple
import numpy as np
import soundfile as sf
import streamlit as st

from frontend.components.charts import plot_noise_degradation, plot_waveform
from frontend.config import NOISE_EVAL_PATH, SNR_LEVELS
from frontend.model_loader import load_noise_evaluation


def add_gaussian_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at a specified SNR (dB) to a 1D audio waveform.

    Standalone implementation for frontend interaction without mutating existing modules.

    Parameters
    ----------
    audio : np.ndarray
        1D float32 audio waveform.
    snr_db : float
        Target Signal-to-Noise Ratio in dB.

    Returns
    -------
    noisy_audio : np.ndarray
        1D float32 audio with added noise.
    """
    audio = np.asarray(audio, dtype=np.float32)
    signal_power = np.mean(audio**2)

    if signal_power < 1e-12:
        return audio.copy()

    noise = np.random.normal(0, 1, audio.shape).astype(np.float32)
    noise_power = np.mean(noise**2)
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))

    noise *= np.sqrt(target_noise_power / (noise_power + 1e-12))
    return (audio + noise).astype(np.float32)


def render_noise_robustness_view():
    """Render the Noise Robustness section."""
    st.markdown("### 🔊 Noise Robustness & Signal Degradation")

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🛡️ Robustness Evaluation Protocol</div>
            <p style="color: #94A3B8; margin-bottom: 0;">
                Environmental sounds frequently occur in challenging acoustic environments. 
                This section evaluates model resilience against calibrated Gaussian noise injected at 
                <b>Clean</b>, <b>20 dB</b>, <b>10 dB</b>, and <b>5 dB</b> SNR levels.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive SNR Audio Playground
    st.markdown("#### 🎚️ Interactive Audio Noise Simulator")

    if "current_raw_audio" not in st.session_state:
        st.markdown(
            """
            <div class="status-banner status-info">
                ℹ️ Upload or select an audio file in <b>Home / Audio Upload</b> to test live noise injection and preview corrupted signals.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_snr = st.select_slider(
                "Select Noise Level (SNR in dB):",
                options=[20, 10, 5],
                value=10,
                help="Lower SNR corresponds to stronger, more severe noise interference.",
            )
            apply_noise_btn = st.button("Generate Noisy Audio", use_container_width=True)

        raw_audio = st.session_state["current_raw_audio"]
        sr = st.session_state["current_sr"]

        # Generate noisy audio
        noisy_audio = add_gaussian_noise(raw_audio, snr_db=selected_snr)

        # Audio Players & Waveforms Side-by-Side
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Clean Signal (Original)**")
            fig_clean = plot_waveform(raw_audio, sample_rate=sr, title="Clean Audio Waveform")
            st.plotly_chart(fig_clean, use_container_width=True)

        with c2:
            st.markdown(f"**Noisy Signal ({selected_snr} dB SNR)**")
            fig_noisy = plot_waveform(noisy_audio, sample_rate=sr, title=f"Noisy Audio ({selected_snr} dB SNR)")
            st.plotly_chart(fig_noisy, use_container_width=True)

    # Noise Evaluation Benchmark Results
    st.markdown("#### 📉 Benchmark Noise Degradation Results")

    noise_results = load_noise_evaluation()

    if noise_results is None:
        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid #3B82F6;">
                <div class="card-title" style="color: #93C5FD;">⏳ Noise Benchmark Evaluation Pending</div>
                <p style="color: #CBD5E1; margin-bottom: 8px;">
                    Full test-set noise evaluation curves across 20 dB, 10 dB, and 5 dB SNR will automatically appear here 
                    once <code>noisy_testing.py</code> evaluation is executed by the modeling team.
                </p>
                <div style="font-size: 0.9rem; color: #94A3B8;">
                    <b>Target file:</b> <code>{NOISE_EVAL_PATH}</code><br>
                    Format: JSON mapping model names to accuracies across SNR levels [Clean, 20, 10, 5].
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        snr_levels = noise_results.get("snr_levels", ["Clean", 20, 10, 5])
        baseline_acc = noise_results.get("baseline_cnn", [])
        hybrid_acc = noise_results.get("cnn_bigru", [])

        fig_deg = plot_noise_degradation(snr_levels, baseline_acc, hybrid_acc)
        st.plotly_chart(fig_deg, use_container_width=True)


"""frontend.views.home_upload

View for Home / Audio Upload, File Inspection, and Audio Playback.
"""

from pathlib import Path
import streamlit as st

from frontend.audio_utils import process_audio_for_inference, read_uploaded_audio
from frontend.components.charts import plot_waveform
from frontend.config import URBANSOUND8K_AUDIO_DIR


def render_home_upload_view():
    """Render the Home and Audio Upload section."""
    st.markdown(
        """
        <div class="card">
            <div class="card-title">📖 Project Overview</div>
            <p style="color: #94A3B8; margin-bottom: 0;">
                This system performs automated <b>Environmental Sound Classification</b> on raw audio waveforms 
                from the <b>UrbanSound8K</b> dataset. It processes 1D acoustic signals sampled at <b>22,050 Hz</b> 
                (4.0 seconds / 88,200 samples) across 10 urban sound categories, comparing a baseline <b>1D CNN</b> 
                against an advanced <b>Hybrid CNN-BiGRU</b> network.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎙️ Audio Selection")

    source_mode = st.radio(
        "Choose audio source:",
        options=["Upload Custom WAV", "Select UrbanSound8K Sample"],
        horizontal=True,
    )

    selected_audio_source = None

    if source_mode == "Upload Custom WAV":
        uploaded_file = st.file_uploader(
            "Upload an audio recording (.wav format)",
            type=["wav"],
            help="Audio will be converted to mono and normalized to 4.0 seconds at 22,050 Hz.",
        )
        if uploaded_file is not None:
            selected_audio_source = uploaded_file
    else:
        # Check if local dataset audio is present
        if URBANSOUND8K_AUDIO_DIR.is_dir():
            sample_candidates = list(URBANSOUND8K_AUDIO_DIR.glob("fold*/*.wav"))[:15]
            if sample_candidates:
                sample_options = {p.name: p for p in sample_candidates}
                selected_sample_name = st.selectbox(
                    "Select a sample recording from UrbanSound8K:",
                    options=list(sample_options.keys()),
                )
                selected_audio_source = sample_options[selected_sample_name]
            else:
                st.info("No sample WAV files found in dataset audio directory.")
        else:
            st.info(f"Dataset directory not found at {URBANSOUND8K_AUDIO_DIR}. Please upload a WAV file.")

    if selected_audio_source is not None:
        try:
            raw_audio, sr, metadata = read_uploaded_audio(selected_audio_source)

            # ── STAGE 1: Audio details + playback ───────────────────────────
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">📊 Audio File Specifications</div>
                    <div class="metric-pill-container">
                        <div class="metric-pill">
                            <span class="metric-pill-label">Filename</span>
                            <span class="metric-pill-val" style="font-size: 0.95rem; word-break: break-all;">{metadata['file_name']}</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-pill-label">Duration</span>
                            <span class="metric-pill-val">{metadata['duration_sec']:.2f} s</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-pill-label">Sampling Rate</span>
                            <span class="metric-pill-val">{metadata['original_sr']:,} Hz</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-pill-label">Channels</span>
                            <span class="metric-pill-val">{metadata['channels']}</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-pill-label">File Size</span>
                            <span class="metric-pill-val">{metadata['file_size_kb']:.1f} KB</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### 🔊 Audio Playback")
            if hasattr(selected_audio_source, "getvalue"):
                st.audio(selected_audio_source.getvalue(), format="audio/wav")
            else:
                with open(selected_audio_source, "rb") as f:
                    st.audio(f.read(), format="audio/wav")

            st.markdown("---")

            # ── STAGE 1 → 2 trigger ─────────────────────────────────────────
            col1, col2 = st.columns([1, 3])
            with col1:
                analyse_btn = st.button(
                    "⚡ Analyse & Preprocess Audio",
                    type="primary",
                    use_container_width=True,
                    help="Resample to 22,050 Hz, convert to mono, and pad/truncate to exactly 4 seconds.",
                )

            if analyse_btn:
                with st.spinner("Preprocessing audio…"):
                    input_tensor, waveform_1d = process_audio_for_inference(raw_audio, sample_rate=sr)

                # Persist in session state for the Classification tab
                st.session_state["current_audio_source"] = selected_audio_source
                st.session_state["current_raw_audio"] = raw_audio
                st.session_state["current_sr"] = sr
                st.session_state["current_metadata"] = metadata
                st.session_state["current_audio_tensor"] = input_tensor
                st.session_state["current_waveform_1d"] = waveform_1d
                st.session_state["preprocessing_done"] = True

            # ── STAGE 2: post-preprocessing results ─────────────────────────
            if st.session_state.get("preprocessing_done") and "current_audio_tensor" in st.session_state:
                input_tensor = st.session_state["current_audio_tensor"]
                waveform_1d = st.session_state["current_waveform_1d"]

                st.markdown(
                    f"""
                    <div class="status-banner" style="background: rgba(16,185,129,0.12);
                         border-left: 4px solid #10B981; padding: 14px 18px;
                         border-radius: 8px; margin: 12px 0;">
                        ✅ <b>Preprocessing complete.</b>&nbsp;
                        Output tensor shape: <code>{input_tensor.shape}</code>
                        &nbsp;·&nbsp; dtype: <code>float32</code>
                        &nbsp;·&nbsp; 22,050 Hz &nbsp;·&nbsp; 88,200 samples (4.0 s mono)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Preprocessed waveform
                fig = plot_waveform(
                    waveform_1d,
                    sample_rate=22050,
                    title="Preprocessed Waveform — 4.0 s @ 22,050 Hz",
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Call-to-action: navigate to Classification tab ───────────
                st.markdown(
                    """
                    <div class="card" style="border-left: 4px solid #00D2FF; margin-top: 8px;">
                        <div class="card-title" style="color: #00D2FF;">👉 Proceed to Sound Classification</div>
                        <p style="color: #CBD5E1; margin-bottom: 0;">
                            Audio preprocessing is complete. Click the
                            <b style="color: #FFFFFF;">🎯 Sound Classification</b>
                            tab above to run model inference and view predicted class probabilities.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            st.error(f"Error reading or processing audio: {exc}")
    else:
        st.markdown(
            """
            <div class="status-banner status-info">
                ℹ️ Upload a WAV recording or select an UrbanSound8K sample above to begin analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )


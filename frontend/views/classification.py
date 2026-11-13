"""frontend.views.classification

View for Sound Classification, Prediction Probabilities, and Waveform Display.

Three-state UX machine
-----------------------
State A — No audio preprocessed:
    Show a warning directing the user to the Home tab.

State B — Audio preprocessed, model weights missing:
    Acknowledge the completed preprocessing; explain that weights are pending
    and that inference will appear automatically once they are available.

State C — Audio preprocessed AND model weights exist:
    Load the model, run inference, and display predicted class, confidence,
    10-class probability chart, and standardised waveform.
"""

import streamlit as st

from frontend.components.charts import plot_class_probabilities, plot_waveform
from frontend.config import CLASS_ID_TO_NAME, CLASS_METADATA, CLASS_NAMES
from frontend.model_loader import get_available_models, load_model_if_available, run_inference


def render_classification_view() -> None:
    """Render the Sound Classification section."""
    st.markdown("### 🎯 Environmental Sound Classification")

    # ── STATE A — no audio in session ───────────────────────────────────────
    if "current_audio_tensor" not in st.session_state:
        st.markdown(
            """
            <div class="status-banner status-warning">
                ⚠️ <b>No audio preprocessed yet.</b>
                Please go to the <b>🏠 Home / Audio Upload</b> tab, select or upload a WAV file,
                then click <b>⚡ Analyse &amp; Preprocess Audio</b> before returning here.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Model selector (always visible once audio is ready) ─────────────────
    models_info = get_available_models()
    col_sel, col_desc = st.columns([1, 2])

    with col_sel:
        selected_model_name = st.selectbox(
            "Select Architecture for Inference:",
            options=list(models_info.keys()),
            index=0,
        )

    model_meta = models_info[selected_model_name]

    with col_desc:
        status_color = "#10B981" if model_meta["exists"] else "#F59E0B"
        status_label = "✅ Weights loaded" if model_meta["exists"] else "⏳ Weights pending"
        st.markdown(
            f"""
            <div style="padding-top: 10px; color: #94A3B8; font-size: 0.9rem;">
                <b>Architecture:</b> {model_meta['description']}<br>
                <b>Weights path:</b> <code>{model_meta['path']}</code><br>
                <span style="color: {status_color}; font-weight: 600;">{status_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── STATE B — audio ready but weights missing ────────────────────────────
    if not model_meta["exists"]:
        input_tensor = st.session_state["current_audio_tensor"]
        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid #10B981; margin-bottom: 16px;">
                <div class="card-title" style="color: #10B981;">✅ Audio Preprocessing Complete</div>
                <p style="color: #CBD5E1; margin-bottom: 0;">
                    Your audio has been standardised to shape
                    <code>{input_tensor.shape}</code> (float32, 22,050 Hz, 4 s mono)
                    and is ready for model inference.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid #F59E0B;">
                <div class="card-title" style="color: #FCD34D;">⏳ Model Weights Pending</div>
                <p style="color: #CBD5E1; margin-bottom: 8px;">
                    Trained weights for <b>{selected_model_name}</b> are not yet available at:
                </p>
                <code style="color: #00D2FF;">{model_meta['path']}</code>
                <p style="color: #94A3B8; margin-top: 12px; margin-bottom: 0; font-size: 0.9rem;">
                    <i>The ML modelling team is currently training the 1D CNN and CNN-BiGRU models.
                    Once weights are placed in <code>saved_models/</code>, live softmax predictions
                    and confidence scores will appear here automatically — no page reload required,
                    just re-select the model.</i>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🎧 Preprocessed Audio Signal")
        waveform_1d = st.session_state.get("current_waveform_1d")
        if waveform_1d is not None:
            fig_wave = plot_waveform(
                waveform_1d,
                sample_rate=22050,
                title="Standardised Waveform — 88,200 Samples (4.0 s @ 22,050 Hz)",
            )
            st.plotly_chart(fig_wave, use_container_width=True)
        return

    # ── STATE C — audio ready AND weights available ──────────────────────────
    model = load_model_if_available(selected_model_name)

    if model is None:
        st.error(
            f"Failed to load model weights from `{model_meta['path']}`. "
            "The file may be corrupted or incompatible. Please check the saved model."
        )
        return

    input_tensor = st.session_state["current_audio_tensor"]
    inference_result = run_inference(model, input_tensor)

    if inference_result is None:
        st.error(
            "Inference returned no result. "
            "Please verify the model output shape (expected 10 softmax values)."
        )
        return

    pred_id, confidence, probs = inference_result
    pred_name = CLASS_ID_TO_NAME.get(pred_id, f"Class {pred_id}")
    class_meta = CLASS_METADATA.get(pred_name, {"icon": "🔊", "category": "Urban Sound"})

    # Prediction Summary Card
    st.markdown(
        f"""
        <div class="card" style="background: linear-gradient(135deg,
             rgba(0,210,255,0.08) 0%, rgba(121,40,202,0.12) 100%);
             margin-bottom: 16px;">
            <div style="font-size: 0.85rem; color: #94A3B8;
                 text-transform: uppercase; letter-spacing: 0.5px;">Predicted Class</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #FFFFFF;
                 display: flex; align-items: center; gap: 12px; margin-top: 4px;">
                <span>{class_meta['icon']}</span>
                <span style="background: linear-gradient(90deg, #00D2FF, #3B82F6);
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {pred_name.replace('_', ' ').title()}
                </span>
            </div>
            <div style="display: flex; gap: 20px; margin-top: 12px;
                 font-size: 0.95rem; color: #CBD5E1;">
                <div><b>Confidence:</b>
                    <span style="color: #00D2FF; font-weight: 700;">{confidence * 100:.2f}%</span>
                </div>
                <div><b>Category:</b> {class_meta['category']}</div>
                <div><b>Class ID:</b> {pred_id}</div>
                <div><b>Model:</b> {selected_model_name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability chart + waveform side by side
    c1, c2 = st.columns([1, 1])
    with c1:
        fig_prob = plot_class_probabilities(probs, class_names=CLASS_NAMES, predicted_idx=pred_id)
        st.plotly_chart(fig_prob, use_container_width=True)

    with c2:
        waveform_1d = st.session_state.get("current_waveform_1d")
        if waveform_1d is not None:
            fig_wave = plot_waveform(
                waveform_1d,
                sample_rate=22050,
                title="Standardised Audio Waveform",
            )
            st.plotly_chart(fig_wave, use_container_width=True)

    # Full-width waveform below
    st.markdown("#### 🎧 Standardised Audio Signal")
    waveform_1d = st.session_state.get("current_waveform_1d")
    if waveform_1d is not None:
        fig_raw = plot_waveform(waveform_1d, title="Preprocessed Waveform (88,200 Samples)")
        st.plotly_chart(fig_raw, use_container_width=True)

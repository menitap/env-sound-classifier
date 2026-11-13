"""app.py

Main Streamlit Application for Environmental Sound Classification.
Features:
1. Home / Audio Upload
2. Sound Classification
3. Model Comparison
4. Noise Robustness
"""

import streamlit as st

from frontend.styles import apply_theme
from frontend.views.classification import render_classification_view
from frontend.views.comparison import render_comparison_view
from frontend.views.home_upload import render_home_upload_view
from frontend.views.noise_robustness import render_noise_robustness_view

# Configure Streamlit page
st.set_page_config(
    page_title="UrbanSound8K Audio Classifier",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply modern dark theme
apply_theme()


def main():
    # Header Banner
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">🔊 Environmental Sound Classifier</h1>
            <div class="app-subtitle">
                UrbanSound8K Classification & Benchmark • Baseline 1D CNN vs. Hybrid 1D CNN-BiGRU
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Tabs
    tabs = st.tabs(
        [
            "🏠 Home / Audio Upload",
            "🎯 Sound Classification",
            "⚖️ Model Comparison",
            "🛡️ Noise Robustness",
        ]
    )

    with tabs[0]:
        render_home_upload_view()

    with tabs[1]:
        render_classification_view()

    with tabs[2]:
        render_comparison_view()

    with tabs[3]:
        render_noise_robustness_view()

    # Sidebar Information
    with st.sidebar:
        st.markdown("### ℹ️ About Project")
        st.markdown(
            """
            **Dataset**: UrbanSound8K  
            **Target Classes**: 10  
            **Sample Rate**: 22,050 Hz  
            **Duration**: 4.0 seconds (88,200 samples)  
            **Channels**: Mono (1D raw waveform)  
            
            ---
            **Models**:
            - 🔹 Baseline 1D CNN
            - 🚀 Hybrid 1D CNN-BiGRU
            
            ---
            **Noise Robustness**:
            - SNR Levels: 20 dB, 10 dB, 5 dB
            """
        )
        st.markdown("---")
        st.caption("Environmental Sound Classification System • Member 4 Frontend")


if __name__ == "__main__":
    main()


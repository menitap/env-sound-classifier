"""frontend.styles

Modern dark-theme styling, CSS variables, and layout aesthetics.
"""

DARK_THEME_CSS = """
<style>
/* Base typography & backgrounds */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}

/* Header gradient */
.app-header {
    background: linear-gradient(135deg, rgba(0, 210, 255, 0.08) 0%, rgba(121, 40, 202, 0.12) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 24px;
}

.app-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00D2FF 0%, #3B82F6 50%, #9333EA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}

.app-subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    margin-top: 6px;
    font-weight: 400;
}

/* Clean Cards */
.card {
    background-color: #1A1D24;
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

.card-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #F8FAFC;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Metric Pill Badges */
.metric-pill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 14px;
}

.metric-pill {
    background: #232730;
    border: 1px solid #333947;
    border-radius: 10px;
    padding: 10px 16px;
    display: flex;
    flex-direction: column;
    min-width: 130px;
}

.metric-pill-label {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}

.metric-pill-val {
    font-size: 1.25rem;
    font-weight: 700;
    color: #00D2FF;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}

/* Status Cards */
.status-banner {
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 16px;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 12px;
}

.status-info {
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #93C5FD;
}

.status-warning {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #FCD34D;
}

.status-success {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6EE7B7;
}

/* Button aesthetics */
div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(0, 210, 255, 0.25);
}
</style>
"""


def apply_theme():
    """Inject custom dark-theme CSS into the Streamlit session."""
    import streamlit as st

    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


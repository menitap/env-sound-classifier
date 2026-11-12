"""frontend.components.charts

Plotly visualizers styled for the dark theme.
"""

from typing import List, Optional, Union
import numpy as np
import plotly.graph_objects as go

from frontend.config import CLASS_NAMES, COLORS


def plot_waveform(
    waveform: np.ndarray,
    sample_rate: int = 22050,
    title: str = "Audio Waveform",
) -> go.Figure:
    """Create an interactive Plotly waveform plot.

    Parameters
    ----------
    waveform : np.ndarray
        1D audio array.
    sample_rate : int
        Audio sampling rate.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
        Configured Plotly figure.
    """
    duration = len(waveform) / sample_rate
    time_axis = np.linspace(0, duration, len(waveform), endpoint=False)

    # Subsample for smooth rendering if array is very large
    step = max(1, len(waveform) // 4000)
    t_sub = time_axis[::step]
    w_sub = waveform[::step]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_sub,
            y=w_sub,
            mode="lines",
            line=dict(color=COLORS["primary"], width=1.2),
            hoverinfo="x+y",
            name="Amplitude",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=16)),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        xaxis=dict(
            title="Time (seconds)",
            color=COLORS["text_muted"],
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
        ),
        yaxis=dict(
            title="Amplitude",
            color=COLORS["text_muted"],
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            range=[-1.05, 1.05],
        ),
        margin=dict(l=40, r=20, t=45, b=40),
        height=260,
    )
    return fig


def plot_class_probabilities(
    probabilities: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    predicted_idx: Optional[int] = None,
) -> go.Figure:
    """Create an interactive horizontal bar chart for 10-class probability distribution.

    Parameters
    ----------
    probabilities : np.ndarray
        Array of 10 class probabilities summing to 1.
    class_names : list of str
        Class names.
    predicted_idx : int, optional
        Index of the winning class to highlight.

    Returns
    -------
    go.Figure
    """
    bar_colors = []
    for i in range(len(class_names)):
        if predicted_idx is not None and i == predicted_idx:
            bar_colors.append(COLORS["accent"])
        else:
            bar_colors.append("rgba(0, 210, 255, 0.35)")

    fig = go.Figure(
        go.Bar(
            x=probabilities * 100.0,
            y=class_names,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="#00D2FF", width=1)),
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=11),
        )
    )

    fig.update_layout(
        title=dict(
            text="Class Probability Distribution",
            font=dict(color=COLORS["text"], size=16),
        ),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        xaxis=dict(
            title="Confidence (%)",
            color=COLORS["text_muted"],
            gridcolor=COLORS["border"],
            range=[0, 105],
        ),
        yaxis=dict(
            color=COLORS["text"],
            autorange="reversed",
            tickfont=dict(size=12),
        ),
        margin=dict(l=140, r=30, t=45, b=40),
        height=380,
    )
    return fig


def plot_noise_degradation(
    snr_levels: List[Union[str, int]],
    baseline_acc: List[Optional[float]],
    hybrid_acc: List[Optional[float]],
) -> go.Figure:
    """Plot accuracy degradation curves across SNR levels."""
    fig = go.Figure()

    # Filter out None/null values gracefully so missing data isn't plotted as 0.0
    b_x, b_y = [], []
    for s, a in zip(snr_levels, baseline_acc or []):
        if a is not None:
            b_x.append(str(s))
            b_y.append(a * 100.0)

    h_x, h_y = [], []
    for s, a in zip(snr_levels, hybrid_acc or []):
        if a is not None:
            h_x.append(str(s))
            h_y.append(a * 100.0)

    fig.add_trace(
        go.Scatter(
            x=b_x,
            y=b_y,
            mode="lines+markers",
            name="Baseline 1D CNN",
            line=dict(color=COLORS["primary"], width=2.5),
            marker=dict(size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=h_x,
            y=h_y,
            mode="lines+markers",
            name="Hybrid CNN-BiGRU",
            line=dict(color=COLORS["secondary"], width=2.5),
            marker=dict(size=8),
        )
    )

    fig.update_layout(
        title=dict(
            text="Accuracy Degradation under Noise",
            font=dict(color=COLORS["text"], size=16),
        ),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        xaxis=dict(
            title="Noise Condition (SNR dB)",
            color=COLORS["text_muted"],
            gridcolor=COLORS["border"],
        ),
        yaxis=dict(
            title="Accuracy (%)",
            color=COLORS["text_muted"],
            gridcolor=COLORS["border"],
            range=[0, 105],
        ),
        margin=dict(l=40, r=30, t=45, b=40),
        height=320,
    )
    return fig

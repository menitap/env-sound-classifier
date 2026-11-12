"""frontend.views.comparison

View for Model Comparison: Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
"""

import plotly.graph_objects as go
import streamlit as st

from frontend.config import CLASS_NAMES, COLORS, METRICS_PATH
from frontend.model_loader import load_confusion_matrices, load_evaluation_metrics


def render_comparison_view():
    """Render the Model Comparison section."""
    st.markdown("### ⚖️ Model Architecture Comparison")

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🔍 Comparative Study</div>
            <p style="color: #94A3B8; margin-bottom: 0;">
                Comparing the <b>Baseline 1D CNN</b> against the proposed <b>Hybrid 1D CNN-BiGRU</b> architecture 
                evaluated on test fold 10 of the UrbanSound8K benchmark.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_data = load_evaluation_metrics()
    cm_data = load_confusion_matrices()

    if metrics_data is None:
        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid #3B82F6;">
                <div class="card-title" style="color: #93C5FD;">📈 Evaluation Metrics Pending</div>
                <p style="color: #CBD5E1; margin-bottom: 8px;">
                    Model evaluation results will populate here once training and test set evaluations are complete.
                </p>
                <div style="font-size: 0.9rem; color: #94A3B8;">
                    <b>Target file:</b> <code>{METRICS_PATH}</code><br>
                    Once Member 3 exports the evaluation results, comparative metrics (Accuracy, Precision, Recall, F1-score) 
                    and interactive confusion matrices will be dynamically visualized.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Architectural specification comparison table
        st.markdown("#### 📐 Architecture Specifications")
        st.table(
            [
                {
                    "Specification": "Input Shape",
                    "Baseline 1D CNN": "(88200, 1)",
                    "Hybrid 1D CNN-BiGRU": "(88200, 1)",
                },
                {
                    "Specification": "Feature Extractor",
                    "Baseline 1D CNN": "3x Conv1D + BatchNorm + MaxPool",
                    "Hybrid 1D CNN-BiGRU": "3x Conv1D + BatchNorm + MaxPool",
                },
                {
                    "Specification": "Temporal Modeling",
                    "Baseline 1D CNN": "Global Average Pooling 1D",
                    "Hybrid 1D CNN-BiGRU": "Bidirectional GRU (64 units)",
                },
                {
                    "Specification": "Dense Classifier",
                    "Baseline 1D CNN": "Dense(128) + Dropout(0.4) + Dense(10)",
                    "Hybrid 1D CNN-BiGRU": "Dense(128) + Dropout(0.4) + Dense(10)",
                },
                {
                    "Specification": "Optimizer / Loss",
                    "Baseline 1D CNN": "Adam (lr=0.001) / Sparse Categorical CE",
                    "Hybrid 1D CNN-BiGRU": "Adam (lr=0.001) / Sparse Categorical CE",
                },
            ]
        )
        return

    # Render actual metrics if present
    baseline_metrics = metrics_data.get("baseline_cnn", {})
    hybrid_metrics = metrics_data.get("cnn_bigru", {})

    cols = st.columns(4)
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    metric_titles = ["Accuracy", "Precision", "Recall", "F1-Score"]

    for i, (m_key, m_title) in enumerate(zip(metric_names, metric_titles)):
        b_val = baseline_metrics.get(m_key, 0.0)
        h_val = hybrid_metrics.get(m_key, 0.0)
        delta = (h_val - b_val) * 100.0

        with cols[i]:
            st.metric(
                label=f"Hybrid {m_title}",
                value=f"{h_val * 100:.2f}%",
                delta=f"{delta:+.2f}% vs CNN" if delta != 0 else None,
            )

    # Metric Comparison Bar Chart
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            name="Baseline 1D CNN",
            x=metric_titles,
            y=[baseline_metrics.get(k, 0.0) * 100 for k in metric_names],
            marker_color=COLORS["primary"],
        )
    )
    fig_bar.add_trace(
        go.Bar(
            name="Hybrid CNN-BiGRU",
            x=metric_titles,
            y=[hybrid_metrics.get(k, 0.0) * 100 for k in metric_names],
            marker_color=COLORS["secondary"],
        )
    )

    fig_bar.update_layout(
        title="Performance Metrics Comparison (%)",
        barmode="group",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        yaxis=dict(range=[0, 105], gridcolor=COLORS["border"]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Confusion Matrices if available
    if cm_data is not None:
        st.markdown("#### 🔢 Confusion Matrices")
        tab1, tab2 = st.tabs(["Hybrid CNN-BiGRU", "Baseline 1D CNN"])

        for tab, m_name in [(tab1, "cnn_bigru"), (tab2, "baseline_cnn")]:
            with tab:
                if m_name in cm_data:
                    cm = cm_data[m_name]
                    fig_cm = go.Figure(
                        data=go.Heatmap(
                            z=cm,
                            x=CLASS_NAMES,
                            y=CLASS_NAMES,
                            colorscale="Viridis",
                        )
                    )
                    fig_cm.update_layout(
                        title=f"Confusion Matrix: {m_name.replace('_', ' ').title()}",
                        xaxis_title="Predicted Label",
                        yaxis_title="True Label",
                        paper_bgcolor=COLORS["surface"],
                        plot_bgcolor=COLORS["surface"],
                        font=dict(color=COLORS["text"]),
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)


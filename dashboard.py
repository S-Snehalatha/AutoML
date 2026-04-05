"""Module 10: Dashboard - analytics overview."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK = "plotly_dark"
BG = "#1A1A2E"
PLOT_BG = "#0D0D1A"


class Dashboard:
    def render(self, session_state):
        st.markdown("## 📊 Analytics Dashboard")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        data = session_state.get("data")
        models = session_state.get("trained_models")
        best = session_state.get("best_model")

        if data is not None:
            c1.metric("Dataset Rows", f"{len(data):,}")
            c2.metric("Features", len(data.columns))
        if models:
            c3.metric("Models Trained", len(models))
        if best:
            c4.metric("Best Model", best.get("name", "N/A"))

        # Leaderboard
        if session_state.get("leaderboard"):
            st.markdown("### 🏆 Model Leaderboard")
            lb_df = pd.DataFrame(session_state["leaderboard"])
            lb_df = lb_df.sort_values(lb_df.columns[1], ascending=False).reset_index(drop=True)
            lb_df.index += 1

            fig = px.bar(lb_df, x="Model", y=lb_df.columns[1],
                         title="Model Comparison",
                         color=lb_df.columns[1],
                         color_continuous_scale=["#6C63FF", "#43E97B"],
                         template=DARK)
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
            st.plotly_chart(fig, use_container_width=True)

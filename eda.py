"""
Module 2: Automated EDA
Generates all visualizations automatically.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

DARK_TEMPLATE = "plotly_dark"
COLOR_SEQ = px.colors.qualitative.Vivid
GRAD_COLORS = ["#6C63FF", "#43E97B", "#FF6584", "#FFD700", "#FF8C42", "#A8DADC"]


class AutoEDA:
    """Auto-generates EDA visualizations for any dataset."""

    def run(self, df: pd.DataFrame, target: str = None) -> dict:
        figs = {}
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if target and target in num_cols:
            num_cols = [c for c in num_cols if c != target]
        if target and target in cat_cols:
            cat_cols = [c for c in cat_cols if c != target]

        # 1. Histograms & distributions
        figs["histograms"] = self._histograms(df, num_cols[:10], target)

        # 2. Box plots
        figs["boxplots"] = self._boxplots(df, num_cols[:10], target)

        # 3. Correlation heatmap
        if len(num_cols) >= 2:
            figs["correlation"] = self._correlation_heatmap(df, num_cols[:20])

        # 4. Scatter plots (target vs numeric)
        if target and target in df.columns:
            figs["scatter"] = self._scatter_plots(df, num_cols[:6], target)
        elif len(num_cols) >= 2:
            figs["scatter"] = self._scatter_plots(df, num_cols[:4], num_cols[0])

        # 5. Categorical plots
        if cat_cols:
            figs["categorical"] = self._categorical_plots(df, cat_cols[:6], target)

        # 6. Missing value heatmap
        figs["missing"] = self._missing_heatmap(df)

        # 7. Pair plot (first 5 numeric cols)
        if len(num_cols) >= 2:
            figs["pairplot"] = self._pairplot(df, num_cols[:5], target)

        return figs

    # ── plot generators ──────────────────────────────────────

    def _histograms(self, df, cols, target):
        figs = []
        n = len(cols)
        if n == 0:
            return figs

        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=cols,
                            vertical_spacing=0.12,
                            horizontal_spacing=0.08)

        for i, col in enumerate(cols):
            row, col_idx = divmod(i, ncols)
            data = df[col].dropna()
            fig.add_trace(
                go.Histogram(x=data, name=col, marker_color=GRAD_COLORS[i % len(GRAD_COLORS)],
                             opacity=0.8, showlegend=False),
                row=row + 1, col=col_idx + 1
            )

        fig.update_layout(template=DARK_TEMPLATE, height=300 * nrows,
                          title_text="Feature Distributions",
                          paper_bgcolor="#1A1A2E", plot_bgcolor="#0D0D1A")
        figs.append(fig)
        return figs

    def _boxplots(self, df, cols, target):
        figs = []
        if not cols:
            return figs

        for col in cols[:6]:
            if target and target in df.columns and df[target].dtype == object:
                fig = px.box(df, x=target, y=col, color=target,
                             title=f"Box Plot: {col} by {target}",
                             color_discrete_sequence=COLOR_SEQ,
                             template=DARK_TEMPLATE)
            else:
                fig = px.box(df, y=col, title=f"Box Plot: {col}",
                             color_discrete_sequence=GRAD_COLORS,
                             template=DARK_TEMPLATE)
            fig.update_layout(paper_bgcolor="#1A1A2E", plot_bgcolor="#0D0D1A")
            figs.append(fig)
        return figs

    def _correlation_heatmap(self, df, cols):
        corr = df[cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig.update_layout(
            title="Correlation Heatmap",
            template=DARK_TEMPLATE,
            height=600,
            paper_bgcolor="#1A1A2E",
            plot_bgcolor="#0D0D1A",
        )
        return fig

    def _scatter_plots(self, df, cols, target):
        figs = []
        for col in cols[:6]:
            if col == target:
                continue
            color_arg = target if target and target in df.columns else None
            try:
                fig = px.scatter(
                    df, x=col, y=target,
                    color=color_arg,
                    title=f"Scatter: {col} vs {target}",
                    trendline="ols",
                    color_discrete_sequence=COLOR_SEQ,
                    template=DARK_TEMPLATE,
                    opacity=0.7,
                )
            except Exception:
                fig = px.scatter(
                    df, x=col, y=target,
                    title=f"Scatter: {col} vs {target}",
                    color_discrete_sequence=COLOR_SEQ,
                    template=DARK_TEMPLATE,
                    opacity=0.7,
                )
            fig.update_layout(paper_bgcolor="#1A1A2E", plot_bgcolor="#0D0D1A")
            figs.append(fig)
        return figs

    def _categorical_plots(self, df, cols, target):
        figs = []
        for col in cols:
            vc = df[col].value_counts().head(20)
            fig = px.bar(
                x=vc.index, y=vc.values,
                title=f"Value Counts: {col}",
                labels={"x": col, "y": "Count"},
                color=vc.values,
                color_continuous_scale=["#6C63FF", "#43E97B"],
                template=DARK_TEMPLATE,
            )
            fig.update_layout(paper_bgcolor="#1A1A2E", plot_bgcolor="#0D0D1A")
            figs.append(fig)

            if target and target in df.columns and df[target].nunique() <= 10:
                cross = pd.crosstab(df[col], df[target])
                if not cross.empty:
                    fig2 = px.bar(
                        cross, barmode="group",
                        title=f"{col} vs {target}",
                        color_discrete_sequence=COLOR_SEQ,
                        template=DARK_TEMPLATE,
                    )
                    fig2.update_layout(paper_bgcolor="#1A1A2E", plot_bgcolor="#0D0D1A")
                    figs.append(fig2)
        return figs

    def _missing_heatmap(self, df):
        missing = df.isnull().astype(int)
        if missing.sum().sum() == 0:
            fig = go.Figure()
            fig.add_annotation(text="✅ No missing values detected!",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=18, color="#43E97B"))
            fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor="#1A1A2E",
                              title="Missing Value Heatmap")
            return fig

        sample = missing.sample(min(200, len(missing)))
        fig = px.imshow(
            sample.T,
            title="Missing Value Heatmap (sample of 200 rows)",
            color_continuous_scale=["#1A1A2E", "#FF6584"],
            template=DARK_TEMPLATE,
            aspect="auto",
        )
        fig.update_layout(paper_bgcolor="#1A1A2E", plot_bgcolor="#0D0D1A", height=400)
        return fig

    def _pairplot(self, df, cols, target):
        plot_df = df[cols + ([target] if target and target in df.columns else [])].dropna().sample(
            min(500, len(df))
        )
        color_col = target if target and target in plot_df.columns else None

        try:
            fig = px.scatter_matrix(
                plot_df,
                dimensions=cols,
                color=color_col,
                title="Pair Plot",
                color_discrete_sequence=COLOR_SEQ,
                template=DARK_TEMPLATE,
                opacity=0.6,
            )
            fig.update_traces(diagonal_visible=True, showupperhalf=False)
            fig.update_layout(height=700, paper_bgcolor="#1A1A2E", plot_bgcolor="#0D0D1A")
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Pair plot unavailable: {e}",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template=DARK_TEMPLATE, paper_bgcolor="#1A1A2E")
        return fig

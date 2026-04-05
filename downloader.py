"""
Module: Downloader
Handles downloading of trained models, predictions, reports, and EDA data.
"""

import io
import json
import pickle
import base64
import datetime
import pandas as pd
import numpy as np
import streamlit as st


def _now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────
# 1. Download trained model as .pkl
# ─────────────────────────────────────────────
def download_model(model_dict: dict, model_name: str = "best_model"):
    """Serialize and download the trained model as a .pkl file."""
    if model_dict is None:
        st.warning("No trained model found. Please train models first.")
        return

    try:
        buffer = io.BytesIO()
        pickle.dump(model_dict["model"], buffer)
        buffer.seek(0)

        filename = f"{model_name.replace(' ', '_')}_{_now()}.pkl"
        st.download_button(
            label="⬇️ Download Model (.pkl)",
            data=buffer,
            file_name=filename,
            mime="application/octet-stream",
            key=f"dl_model_{_now()}",
            help="Downloads the trained scikit-learn compatible model as a pickle file.",
        )
        st.caption(f"📦 Model: **{model_dict.get('name', model_name)}** | "
                   f"Metrics: {model_dict.get('metrics', {})}")
    except Exception as e:
        st.error(f"Failed to serialize model: {e}")


# ─────────────────────────────────────────────
# 2. Download full pipeline (model + preprocessor) as .pkl
# ─────────────────────────────────────────────
def download_pipeline(model_dict: dict, preprocessor, model_name: str = "pipeline"):
    """Download both the model and preprocessor bundled together."""
    if model_dict is None or preprocessor is None:
        st.warning("Model or preprocessor not found. Complete training first.")
        return

    try:
        pipeline_bundle = {
            "model": model_dict["model"],
            "preprocessor": preprocessor,
            "model_name": model_dict.get("name", model_name),
            "metrics": model_dict.get("metrics", {}),
            "exported_at": _now(),
        }
        buffer = io.BytesIO()
        pickle.dump(pipeline_bundle, buffer)
        buffer.seek(0)

        filename = f"pipeline_{model_name.replace(' ', '_')}_{_now()}.pkl"
        st.download_button(
            label="⬇️ Download Full Pipeline (.pkl)",
            data=buffer,
            file_name=filename,
            mime="application/octet-stream",
            key=f"dl_pipeline_{_now()}",
            help="Downloads model + preprocessor bundle. Load with pickle.load() to make predictions.",
        )
        st.caption("💡 Load with: `import pickle; bundle = pickle.load(open('pipeline.pkl','rb'))`")
    except Exception as e:
        st.error(f"Failed to bundle pipeline: {e}")


# ─────────────────────────────────────────────
# 3. Download predictions as CSV
# ─────────────────────────────────────────────
def download_predictions(predictions, input_df: pd.DataFrame = None,
                          target_col: str = "Prediction"):
    """Download predictions as a CSV file."""
    if predictions is None:
        st.warning("No predictions available yet.")
        return

    try:
        if input_df is not None:
            result_df = input_df.copy()
            result_df[target_col] = predictions
        else:
            result_df = pd.DataFrame({target_col: predictions})

        csv_data = result_df.to_csv(index=False)
        filename = f"predictions_{_now()}.csv"

        st.download_button(
            label="⬇️ Download Predictions (.csv)",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"dl_preds_{_now()}",
            help="Downloads input features with predicted values appended as a new column.",
        )
        st.caption(f"📊 {len(result_df)} rows × {len(result_df.columns)} columns")
    except Exception as e:
        st.error(f"Failed to export predictions: {e}")


# ─────────────────────────────────────────────
# 4. Download model leaderboard as CSV
# ─────────────────────────────────────────────
def download_leaderboard(leaderboard: list):
    """Download the model comparison leaderboard as CSV."""
    if not leaderboard:
        st.warning("No leaderboard data. Train models first.")
        return

    try:
        lb_df = pd.DataFrame(leaderboard)
        csv_data = lb_df.to_csv(index=False)
        filename = f"leaderboard_{_now()}.csv"

        st.download_button(
            label="⬇️ Download Leaderboard (.csv)",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"dl_leaderboard_{_now()}",
            help="Downloads all trained model names and their evaluation metrics.",
        )
    except Exception as e:
        st.error(f"Failed to export leaderboard: {e}")


# ─────────────────────────────────────────────
# 5. Download evaluation metrics as JSON
# ─────────────────────────────────────────────
def download_evaluation_report(evaluation_results: dict, model_name: str = "model"):
    """Download evaluation metrics as a structured JSON report."""
    if not evaluation_results:
        st.warning("No evaluation results yet.")
        return

    try:
        report = {
            "model_name": model_name,
            "exported_at": _now(),
            "metrics": evaluation_results.get("metrics", {}),
        }
        json_str = json.dumps(report, indent=2, default=str)
        filename = f"eval_report_{model_name.replace(' ', '_')}_{_now()}.json"

        st.download_button(
            label="⬇️ Download Evaluation Report (.json)",
            data=json_str,
            file_name=filename,
            mime="application/json",
            key=f"dl_eval_{_now()}",
            help="Downloads all evaluation metrics in JSON format.",
        )
    except Exception as e:
        st.error(f"Failed to export report: {e}")


# ─────────────────────────────────────────────
# 6. Download preprocessed dataset as CSV
# ─────────────────────────────────────────────
def download_preprocessed_data(X: np.ndarray, y, feature_names: list,
                                 target_col: str = "target"):
    """Download the cleaned, preprocessed dataset."""
    if X is None:
        st.warning("No preprocessed data found.")
        return

    try:
        df = pd.DataFrame(X, columns=feature_names)
        if y is not None:
            df[target_col] = y
        csv_data = df.to_csv(index=False)
        filename = f"preprocessed_data_{_now()}.csv"

        st.download_button(
            label="⬇️ Download Preprocessed Data (.csv)",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"dl_preprocessed_{_now()}",
            help="Downloads the fully preprocessed (imputed, encoded, scaled) dataset.",
        )
        st.caption(f"📋 {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Failed to export preprocessed data: {e}")


# ─────────────────────────────────────────────
# 7. Download all plots as HTML (interactive)
# ─────────────────────────────────────────────
def download_eda_report(eda_figures: dict):
    """Bundle all EDA figures into a single interactive HTML report."""
    if not eda_figures:
        st.warning("No EDA figures found. Run EDA first.")
        return

    try:
        html_parts = ["""
        <html><head>
        <title>AutoML Studio — EDA Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
          body { background:#0d0d1a; color:#f0f0ff; font-family:'Segoe UI',sans-serif; padding:20px; }
          h1 { color:#6C63FF; } h2 { color:#43E97B; border-bottom:1px solid #333; padding-bottom:6px; }
          .fig-wrap { background:#1a1a2e; border-radius:12px; padding:16px; margin:20px 0; }
        </style></head><body>
        <h1>⚡ AutoML Studio — EDA Report</h1>
        """]

        for section_name, figs in eda_figures.items():
            html_parts.append(f"<h2>{section_name}</h2>")
            if isinstance(figs, list):
                for fig in figs:
                    if fig is not None:
                        html_parts.append(
                            f'<div class="fig-wrap">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
                        )
            elif figs is not None:
                html_parts.append(
                    f'<div class="fig-wrap">{figs.to_html(full_html=False, include_plotlyjs=False)}</div>'
                )

        html_parts.append("</body></html>")
        html_str = "\n".join(html_parts)
        filename = f"eda_report_{_now()}.html"

        st.download_button(
            label="⬇️ Download EDA Report (.html)",
            data=html_str,
            file_name=filename,
            mime="text/html",
            key=f"dl_eda_{_now()}",
            help="Downloads all EDA charts as an interactive HTML file you can open in any browser.",
        )
    except Exception as e:
        st.error(f"Failed to generate EDA report: {e}")


# ─────────────────────────────────────────────
# 8. Download single Plotly figure as HTML
# ─────────────────────────────────────────────
def download_figure(fig, label: str = "chart", button_label: str = "⬇️ Download Chart"):
    """Download any single Plotly figure as an interactive HTML file."""
    if fig is None:
        return
    try:
        html_str = fig.to_html(include_plotlyjs="cdn")
        safe_label = label.replace(" ", "_").replace("-", "_").lower()
        filename = f"{safe_label}.html"
        st.download_button(
            label=button_label,
            data=html_str,
            file_name=filename,
            mime="text/html",
            key=f"dl_fig_{safe_label}",
        )
    except Exception as e:
        st.error(f"Download failed: {e}")


# ─────────────────────────────────────────────
# 9. Master Download Center (renders all buttons)
# ─────────────────────────────────────────────
def render_download_center(session_state):
    """Download center — shows only the trained model download."""
    st.markdown("---")
    st.markdown("## ⬇️ Download Model")

    if not session_state.get("best_model"):
        st.info("💡 Complete the pipeline (upload data → train) to download your model.")
        return

    best = session_state["best_model"]
    model_name = best.get("name", "best_model")

    st.markdown(f"**Best Model:** `{model_name}`")
    metrics = best.get("metrics", {})
    if metrics:
        met_str = " · ".join(
            f"{k}: **{v:.4f}**" if isinstance(v, float) else f"{k}: **{v}**"
            for k, v in metrics.items()
        )
        st.markdown(met_str)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # Model only (.pkl)
        import io, pickle
        try:
            buf = io.BytesIO()
            pickle.dump(best["model"], buf)
            buf.seek(0)
            st.download_button(
                label="⬇️ Download Model (.pkl)",
                data=buf,
                file_name=f"{model_name.replace(' ', '_')}_model.pkl",
                mime="application/octet-stream",
                key="dl_model_only",
            )
            st.caption("Load with: `import pickle; model = pickle.load(open('model.pkl','rb'))`")
        except Exception as e:
            st.error(f"Failed to serialize model: {e}")

    with col2:
        # Full pipeline (model + preprocessor)
        if session_state.get("preprocessor"):
            try:
                bundle = {
                    "model":        best["model"],
                    "preprocessor": session_state["preprocessor"],
                    "model_name":   model_name,
                    "metrics":      metrics,
                }
                buf2 = io.BytesIO()
                pickle.dump(bundle, buf2)
                buf2.seek(0)
                st.download_button(
                    label="⬇️ Download Full Pipeline (.pkl)",
                    data=buf2,
                    file_name=f"{model_name.replace(' ', '_')}_pipeline.pkl",
                    mime="application/octet-stream",
                    key="dl_pipeline_only",
                )
                st.caption("Load with: `bundle = pickle.load(open('pipeline.pkl','rb'))`")
            except Exception as e:
                st.error(f"Failed to bundle pipeline: {e}")
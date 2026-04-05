"""
Module 6 & 7: Model Evaluation + Visual Analysis
All metrics and visual plots
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, roc_auc_score, confusion_matrix,
                               mean_absolute_error, mean_squared_error, r2_score,
                               roc_curve, precision_recall_curve,
                               silhouette_score, classification_report)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, Any, Optional, List


DARK_LAYOUT = dict(template="plotly_dark", paper_bgcolor="#0f172a",
                    plot_bgcolor="#1e293b", font=dict(color="#e2e8f0"))


class ModelEvaluator:
    """Computes metrics and generates visual analysis plots."""

    def __init__(self, task_type: str):
        self.task_type = task_type
        self.results: Dict[str, Dict] = {}

    def _fig_json(self, fig) -> str:
        return fig.to_json()

    def evaluate_classification(self, model_name: str, model,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  labels: Optional[List] = None) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        }
        if y_proba is not None:
            try:
                n_classes = len(np.unique(y_test))
                avg = "binary" if n_classes == 2 else "weighted"
                proba_use = y_proba[:, 1] if n_classes == 2 else y_proba
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba_use,
                                                                 multi_class="ovr", average=avg)), 4)
            except Exception:
                metrics["roc_auc"] = None

        self.results[model_name] = metrics
        return metrics

    def evaluate_regression(self, model_name: str, model,
                              X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        metrics = {
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "mse": round(float(mean_squared_error(y_test, y_pred)), 4),
        }
        self.results[model_name] = metrics
        return metrics

    def evaluate_clustering(self, model_name: str, model,
                              X: np.ndarray) -> Dict[str, Any]:
        labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        metrics = {"n_clusters": n_clusters}
        try:
            if n_clusters > 1:
                metrics["silhouette_score"] = round(float(silhouette_score(X, labels)), 4)
        except Exception:
            pass
        self.results[model_name] = metrics
        return metrics

    # ========== VISUAL ANALYSIS ==========

    def confusion_matrix_plot(self, model, X_test: np.ndarray,
                               y_test: np.ndarray, class_names: Optional[List] = None) -> str:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        labels = class_names or [str(i) for i in sorted(np.unique(y_test))]
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale="Blues", text=cm,
            texttemplate="%{text}", textfont={"size": 14}
        ))
        fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted",
                           yaxis_title="Actual", **DARK_LAYOUT)
        return self._fig_json(fig)

    def roc_curve_plot(self, models_dict: Dict, X_test: np.ndarray,
                        y_test: np.ndarray) -> str:
        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash", color="#64748b"),
                       x0=0, x1=1, y0=0, y1=1)
        colors = px.colors.qualitative.Set1

        for i, (name, model) in enumerate(models_dict.items()):
            if not hasattr(model, "predict_proba"):
                continue
            try:
                proba = model.predict_proba(X_test)
                n_classes = len(np.unique(y_test))
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                    auc = roc_auc_score(y_test, proba[:, 1])
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                             name=f"{name} (AUC={auc:.3f})",
                                             line=dict(color=colors[i % len(colors)], width=2)))
            except Exception:
                continue

        fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate", **DARK_LAYOUT)
        return self._fig_json(fig)

    def precision_recall_plot(self, models_dict: Dict, X_test: np.ndarray,
                               y_test: np.ndarray) -> str:
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, (name, model) in enumerate(models_dict.items()):
            if not hasattr(model, "predict_proba"):
                continue
            try:
                proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    prec, rec, _ = precision_recall_curve(y_test, proba[:, 1])
                    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                             name=name,
                                             line=dict(color=colors[i % len(colors)], width=2)))
            except Exception:
                continue
        fig.update_layout(title="Precision-Recall Curves",
                           xaxis_title="Recall", yaxis_title="Precision", **DARK_LAYOUT)
        return self._fig_json(fig)

    def residual_plot(self, model, X_test: np.ndarray, y_test: np.ndarray) -> str:
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        fig = make_subplots(rows=1, cols=2,
                             subplot_titles=["Residuals vs Fitted", "Residual Distribution"])
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                                  marker=dict(color="#6366f1", opacity=0.5, size=4),
                                  name="Residuals"), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#f43f5e", row=1, col=1)
        fig.add_trace(go.Histogram(x=residuals, nbinsx=40,
                                    marker_color="#8b5cf6", name="Distribution"), row=1, col=2)
        fig.update_layout(title="Residual Analysis", **DARK_LAYOUT)
        return self._fig_json(fig)

    def predicted_vs_actual_plot(self, model, X_test: np.ndarray, y_test: np.ndarray) -> str:
        y_pred = model.predict(X_test)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                  marker=dict(color="#06b6d4", opacity=0.5, size=4),
                                  name="Predictions"))
        line_min, line_max = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[line_min, line_max], y=[line_min, line_max],
                                  mode="lines", line=dict(color="#f43f5e", dash="dash"),
                                  name="Perfect Fit"))
        fig.update_layout(title="Predicted vs Actual", xaxis_title="Actual",
                           yaxis_title="Predicted", **DARK_LAYOUT)
        return self._fig_json(fig)

    def learning_curve_plot(self, model, X: np.ndarray, y: np.ndarray) -> str:
        try:
            sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=3, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy" if self.task_type == "classification" else "r2"
            )
        except Exception:
            return "{}"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sizes, y=train_scores.mean(axis=1), mode="lines+markers",
                                  name="Train Score", line=dict(color="#6366f1", width=2)))
        fig.add_trace(go.Scatter(x=sizes, y=val_scores.mean(axis=1), mode="lines+markers",
                                  name="Validation Score", line=dict(color="#f59e0b", width=2)))
        fig.update_layout(title="Learning Curves", xaxis_title="Training Size",
                           yaxis_title="Score", **DARK_LAYOUT)
        return self._fig_json(fig)

    def model_comparison_chart(self) -> str:
        if not self.results:
            return "{}"
        names = list(self.results.keys())
        if self.task_type == "classification":
            metric_key = "accuracy"
        elif self.task_type == "regression":
            metric_key = "r2"
        else:
            metric_key = "silhouette_score"

        values = [self.results[n].get(metric_key, 0) for n in names]
        fig = px.bar(x=names, y=values, title=f"Model Comparison ({metric_key})",
                     color=values, color_continuous_scale="Viridis",
                     labels={"x": "Model", "y": metric_key})
        fig.update_layout(**DARK_LAYOUT)
        return self._fig_json(fig)

    def lift_chart(self, model, X_test: np.ndarray, y_test: np.ndarray) -> str:
        if not hasattr(model, "predict_proba") or len(np.unique(y_test)) != 2:
            return "{}"
        proba = model.predict_proba(X_test)[:, 1]
        df = pd.DataFrame({"y": y_test, "proba": proba})
        df = df.sort_values("proba", ascending=False).reset_index(drop=True)
        df["cumulative_pos"] = df["y"].cumsum()
        df["pct_pop"] = (df.index + 1) / len(df)
        baseline = df["y"].mean()
        df["lift"] = (df["cumulative_pos"] / (df.index + 1)) / baseline

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["pct_pop"], y=df["lift"], mode="lines",
                                  line=dict(color="#6366f1", width=2), name="Lift"))
        fig.add_hline(y=1, line_dash="dash", line_color="#f43f5e")
        fig.update_layout(title="Lift Chart", xaxis_title="% Population",
                           yaxis_title="Lift", **DARK_LAYOUT)
        return self._fig_json(fig)

    def cumulative_gain_chart(self, model, X_test: np.ndarray, y_test: np.ndarray) -> str:
        if not hasattr(model, "predict_proba") or len(np.unique(y_test)) != 2:
            return "{}"
        proba = model.predict_proba(X_test)[:, 1]
        df = pd.DataFrame({"y": y_test, "proba": proba})
        df = df.sort_values("proba", ascending=False).reset_index(drop=True)
        df["cum_gain"] = df["y"].cumsum() / df["y"].sum()
        df["pct_pop"] = (df.index + 1) / len(df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["pct_pop"], y=df["cum_gain"], mode="lines",
                                  line=dict(color="#10b981", width=2), name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="#64748b"), name="Random"))
        fig.update_layout(title="Cumulative Gain Chart", xaxis_title="% Population",
                           yaxis_title="% Positive Captured", **DARK_LAYOUT)
        return self._fig_json(fig)

    def get_all_results(self) -> Dict[str, Dict]:
        return self.results

    def get_leaderboard(self) -> List[Dict]:
        rows = []
        for name, metrics in self.results.items():
            row = {"model": name}
            row.update(metrics)
            rows.append(row)
        if self.task_type == "classification":
            rows.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
        elif self.task_type == "regression":
            rows.sort(key=lambda x: x.get("r2", -999), reverse=True)
        return rows

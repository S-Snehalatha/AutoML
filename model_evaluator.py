"""
Module 5: Model Evaluator
Full metrics + visual analysis plots.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

DARK = "plotly_dark"
BG = "#1A1A2E"
PLOT_BG = "#0D0D1A"


class ModelEvaluator:
    def __init__(self, task_type="Classification"):
        self.task_type = task_type

    def evaluate(self, model_dict, X, y, feature_names=None):
        model = model_dict["model"]
        result = {"metrics": {}, "figures": {}}

        if self.task_type == "Clustering":
            labels = model.labels_ if hasattr(model, "labels_") else model.fit_predict(X)
            unique = np.unique(labels)
            valid = unique[unique != -1]
            if len(valid) >= 2:
                result["metrics"]["Silhouette Score"] = round(silhouette_score(X, labels), 4)
            result["metrics"]["N Clusters"] = len(valid)
            result["metrics"]["N Noise Points"] = int((labels == -1).sum())
            result["figures"]["Cluster Plot"] = self._cluster_plot(X, labels)
            return result

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        if self.task_type == "Classification":
            result["metrics"] = self._clf_metrics(y_te, y_pred, model, X_te)
            result["figures"] = self._clf_figures(y_te, y_pred, model, X_te, X_tr, y_tr, feature_names)
        else:
            result["metrics"] = self._reg_metrics(y_te, y_pred)
            result["figures"] = self._reg_figures(y_te, y_pred, model, X, y)

        return result

    # ── Classification ──────────────────────────────────────

    def _clf_metrics(self, y_te, y_pred, model, X_te):
        n_classes = len(np.unique(y_te))
        avg = "binary" if n_classes == 2 else "weighted"
        metrics = {
            "Accuracy": round(accuracy_score(y_te, y_pred), 4),
            "Precision": round(precision_score(y_te, y_pred, average=avg, zero_division=0), 4),
            "Recall": round(recall_score(y_te, y_pred, average=avg, zero_division=0), 4),
            "F1 Score": round(f1_score(y_te, y_pred, average=avg, zero_division=0), 4),
        }
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_te)
                if n_classes == 2:
                    metrics["ROC AUC"] = round(roc_auc_score(y_te, proba[:, 1]), 4)
                else:
                    metrics["ROC AUC"] = round(roc_auc_score(y_te, proba, multi_class="ovr", average="weighted"), 4)
        except Exception:
            pass
        return metrics

    def _clf_figures(self, y_te, y_pred, model, X_te, X_tr, y_tr, feature_names):
        figs = {}

        # Confusion Matrix
        cm = confusion_matrix(y_te, y_pred)
        labels = [str(c) for c in sorted(np.unique(y_te))]
        fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                        title="Confusion Matrix", color_continuous_scale="Purples",
                        template=DARK)
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
        figs["Confusion Matrix"] = fig

        # ROC Curve (binary only)
        if hasattr(model, "predict_proba") and len(np.unique(y_te)) == 2:
            try:
                proba = model.predict_proba(X_te)[:, 1]
                fpr, tpr, _ = roc_curve(y_te, proba)
                auc = roc_auc_score(y_te, proba)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"ROC (AUC={auc:.3f})",
                                          line=dict(color="#6C63FF", width=2)))
                fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          name="Random", line=dict(color="gray", dash="dash")))
                fig2.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                                   template=DARK, paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
                figs["ROC Curve"] = fig2

                # PR Curve
                pre, rec, _ = precision_recall_curve(y_te, proba)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=rec, y=pre, mode="lines",
                                          name="Precision-Recall",
                                          line=dict(color="#43E97B", width=2)))
                fig3.update_layout(title="Precision-Recall Curve", template=DARK,
                                   paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
                figs["Precision-Recall Curve"] = fig3

                # Lift Chart
                figs["Lift Chart"] = self._lift_chart(y_te, proba)
                figs["Cumulative Gain"] = self._cumulative_gain(y_te, proba)
            except Exception:
                pass

        # Learning Curves
        try:
            figs["Learning Curves"] = self._learning_curves(model, np.vstack([X_tr, X_te]),
                                                              np.concatenate([y_tr, y_te]))
        except Exception:
            pass

        return figs

    # ── Regression ─────────────────────────────────────────

    def _reg_metrics(self, y_te, y_pred):
        return {
            "MAE": round(mean_absolute_error(y_te, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_te, y_pred)), 4),
            "R² Score": round(r2_score(y_te, y_pred), 4),
        }

    def _reg_figures(self, y_te, y_pred, model, X, y):
        figs = {}

        # Predicted vs Actual
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te, y=y_pred, mode="markers",
                                  name="Predictions",
                                  marker=dict(color="#6C63FF", opacity=0.6)))
        mn = min(y_te.min(), y_pred.min())
        mx = max(y_te.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                  name="Perfect fit", line=dict(color="#43E97B", dash="dash")))
        fig.update_layout(title="Predicted vs Actual", xaxis_title="Actual",
                          yaxis_title="Predicted", template=DARK, paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
        figs["Predicted vs Actual"] = fig

        # Residual Plot
        residuals = y_te - y_pred
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                                   name="Residuals",
                                   marker=dict(color="#FF6584", opacity=0.6)))
        fig2.add_hline(y=0, line_dash="dash", line_color="#43E97B")
        fig2.update_layout(title="Residual Plot", xaxis_title="Fitted Values",
                           yaxis_title="Residuals", template=DARK, paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
        figs["Residual Plot"] = fig2

        # Residual distribution
        fig3 = px.histogram(x=residuals, nbins=40, title="Residual Distribution",
                            color_discrete_sequence=["#6C63FF"], template=DARK)
        fig3.update_layout(paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
        figs["Residual Distribution"] = fig3

        # Learning curves
        try:
            figs["Learning Curves"] = self._learning_curves(model, X, y, scoring="r2")
        except Exception:
            pass

        return figs

    # ── Clustering ─────────────────────────────────────────

    def _cluster_plot(self, X, labels):
        if X.shape[1] >= 2:
            from sklearn.decomposition import PCA
            X_2d = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X
            fig = px.scatter(x=X_2d[:, 0], y=X_2d[:, 1], color=[str(l) for l in labels],
                             title="Cluster Plot (PCA 2D)", template=DARK,
                             color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
            return fig
        return go.Figure()

    # ── Utility plots ───────────────────────────────────────

    def _learning_curves(self, model, X, y, scoring="accuracy"):
        train_sizes = np.linspace(0.1, 1.0, 8)
        try:
            sizes, train_sc, val_sc = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=3,
                scoring=scoring, n_jobs=-1
            )
        except Exception:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sizes, y=train_sc.mean(axis=1),
                                  mode="lines+markers", name="Train",
                                  line=dict(color="#6C63FF")))
        fig.add_trace(go.Scatter(x=sizes, y=val_sc.mean(axis=1),
                                  mode="lines+markers", name="Validation",
                                  line=dict(color="#43E97B")))
        fig.update_layout(title="Learning Curves", xaxis_title="Training size",
                          yaxis_title=scoring, template=DARK, paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
        return fig

    def _lift_chart(self, y_te, proba):
        df = pd.DataFrame({"y": y_te, "p": proba}).sort_values("p", ascending=False).reset_index(drop=True)
        df["decile"] = pd.qcut(df.index, 10, labels=False)
        lift = df.groupby("decile").apply(lambda g: g["y"].mean() / df["y"].mean())
        fig = px.bar(x=list(range(1, 11)), y=lift.values, title="Lift Chart",
                     labels={"x": "Decile", "y": "Lift"},
                     color=lift.values, color_continuous_scale=["#6C63FF", "#43E97B"],
                     template=DARK)
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
        return fig

    def _cumulative_gain(self, y_te, proba):
        df = pd.DataFrame({"y": y_te, "p": proba}).sort_values("p", ascending=False).reset_index(drop=True)
        cum_gain = (df["y"].cumsum() / df["y"].sum()).values
        pct_pop = np.linspace(0, 1, len(cum_gain))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pct_pop, y=cum_gain, mode="lines",
                                  name="Model", line=dict(color="#6C63FF", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  name="Baseline", line=dict(color="gray", dash="dash")))
        fig.update_layout(title="Cumulative Gain Chart", xaxis_title="% Population",
                          yaxis_title="% Positive", template=DARK, paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
        return fig

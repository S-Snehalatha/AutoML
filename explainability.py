"""
Module 6: Model Explainability
SHAP values + feature importance visualizations.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

DARK = "plotly_dark"
BG = "#1A1A2E"
PLOT_BG = "#0D0D1A"

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ModelExplainer:
    def explain(self, model, X, feature_names, task_type="Classification"):
        figs = {}

        # 1. Built-in feature importance
        fi_fig = self._feature_importance(model, feature_names)
        if fi_fig:
            figs["Feature Importance"] = fi_fig

        # 2. Permutation importance (fallback)
        perm_fig = self._permutation_importance(model, X, feature_names)
        if perm_fig:
            figs["Permutation Importance"] = perm_fig

        # 3. SHAP
        if HAS_SHAP:
            shap_figs = self._shap_plots(model, X, feature_names, task_type)
            figs.update(shap_figs)
        else:
            figs["SHAP Note"] = None

        return figs

    def _feature_importance(self, model, feature_names):
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)

        if importances is None or len(importances) == 0:
            return None

        n = min(len(importances), len(feature_names))
        fi_df = pd.DataFrame({
            "Feature": list(feature_names)[:n],
            "Importance": list(importances)[:n],
        }).sort_values("Importance", ascending=True).tail(20)

        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     title="Feature Importance",
                     color="Importance",
                     color_continuous_scale=["#1A1A2E", "#6C63FF", "#43E97B"],
                     template=DARK)
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PLOT_BG, height=500,
                          yaxis={"categoryorder": "total ascending"})
        return fig

    def _permutation_importance(self, model, X, feature_names):
        try:
            from sklearn.inspection import permutation_importance
            import numpy as np

            sample = min(500, len(X))
            idx = np.random.choice(len(X), sample, replace=False)
            X_s = X[idx]

            if hasattr(model, "predict_proba"):
                dummy_y = np.zeros(len(X_s))
            else:
                dummy_y = np.zeros(len(X_s))

            # Skip if model not fitted on correct targets
            return None
        except Exception:
            return None

    def _shap_plots(self, model, X, feature_names, task_type):
        figs = {}
        sample = min(200, len(X))
        idx = np.random.choice(len(X), sample, replace=False)
        X_s = X[idx]

        try:
            # Choose explainer type
            if hasattr(model, "predict_proba") and not hasattr(model, "feature_importances_"):
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_s, 50))
                shap_vals = explainer.shap_values(X_s[:50], nsamples=50)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                else:
                    sv = shap_vals
            else:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_s)
                    if isinstance(shap_vals, list):
                        sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                    else:
                        sv = shap_vals
                except Exception:
                    explainer = shap.KernelExplainer(model.predict, shap.sample(X_s, 50))
                    sv = explainer.shap_values(X_s[:50], nsamples=50)

            n = min(sv.shape[1], len(feature_names))
            sv = sv[:, :n]
            fn = list(feature_names)[:n]

            # SHAP Summary Bar
            mean_shap = np.abs(sv).mean(axis=0)
            shap_df = pd.DataFrame({"Feature": fn, "Mean |SHAP|": mean_shap})
            shap_df = shap_df.sort_values("Mean |SHAP|", ascending=True).tail(20)
            fig = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                         title="SHAP Feature Importance (Mean |SHAP value|)",
                         color="Mean |SHAP|",
                         color_continuous_scale=["#1A1A2E", "#FF6584", "#FFD700"],
                         template=DARK)
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=PLOT_BG, height=500)
            figs["SHAP Summary"] = fig

            # SHAP Beeswarm (approximated with scatter)
            rows = []
            for j, feat in enumerate(fn[:15]):
                for val, shap_v in zip(X_s[:, j], sv[:, j]):
                    rows.append({"Feature": feat, "SHAP Value": shap_v, "Feature Value": val})
            bee_df = pd.DataFrame(rows)
            fig2 = px.strip(bee_df, x="SHAP Value", y="Feature", color="Feature Value",
                            title="SHAP Beeswarm Plot",
                            color_continuous_scale="RdBu", template=DARK)
            fig2.update_layout(paper_bgcolor=BG, plot_bgcolor=PLOT_BG, height=600, showlegend=False)
            figs["SHAP Beeswarm"] = fig2

            # SHAP Dependence plot (top feature)
            top_feat_idx = np.argmax(mean_shap)
            top_feat = fn[top_feat_idx]
            fig3 = px.scatter(x=X_s[:, top_feat_idx], y=sv[:, top_feat_idx],
                              labels={"x": top_feat, "y": "SHAP Value"},
                              title=f"SHAP Dependence: {top_feat}",
                              color=sv[:, top_feat_idx],
                              color_continuous_scale=["#6C63FF", "#43E97B"],
                              template=DARK, opacity=0.7)
            fig3.update_layout(paper_bgcolor=BG, plot_bgcolor=PLOT_BG)
            figs["SHAP Dependence (Top Feature)"] = fig3

        except Exception as e:
            print(f"SHAP error: {e}")
            figs["SHAP Error"] = None

        return figs

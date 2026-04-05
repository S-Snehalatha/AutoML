"""Module 7: Model Selector - picks the best model automatically."""

import numpy as np


class ModelSelector:
    def __init__(self, task_type="Classification"):
        self.task_type = task_type

    def select(self, trained_models, leaderboard):
        if not trained_models:
            return None

        if self.task_type == "Classification":
            primary = "F1 Score"
            secondary = "Accuracy"
        elif self.task_type == "Regression":
            primary = "R² Score"
            secondary = "RMSE"
        else:
            primary = "Silhouette Score"
            secondary = "Clusters"

        best = None
        best_score = -np.inf

        for model_dict in trained_models:
            metrics = model_dict.get("metrics", {})
            score = metrics.get(primary, metrics.get(secondary, 0))
            if score is None:
                score = 0
            # For RMSE, lower is better
            if primary == "RMSE":
                score = -score
            if score > best_score:
                best_score = score
                best = model_dict

        return best

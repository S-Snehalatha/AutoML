"""
Module 5: Hyperparameter Optimization
Grid Search, Random Search, Bayesian Optimization (Optuna)
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
from typing import Dict, Any, Optional, Tuple
import time


PARAM_GRIDS = {
    "classification": {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [50, 100, 200],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "num_leaves": [31, 63, 127],
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"],
        },
    },
    "regression": {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
        },
        "LightGBM": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.3],
            "num_leaves": [31, 63],
        },
    }
}

OPTUNA_PARAMS = {
    "classification": {
        "Random Forest": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        },
        "XGBoost": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        },
        "LightGBM": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        },
    },
    "regression": {
        "Random Forest": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        },
        "XGBoost": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        },
        "LightGBM": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        },
    }
}


class HyperparameterOptimizer:
    """Runs Grid, Random and Bayesian hyperparameter search."""

    def __init__(self, task_type: str, random_state: int = 42):
        self.task_type = task_type
        self.random_state = random_state
        self.results: Dict[str, Any] = {}
        self.scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"

    def grid_search(self, model, model_name: str,
                     X: np.ndarray, y: np.ndarray,
                     cv: int = 3) -> Tuple[Any, Dict]:
        param_grid = PARAM_GRIDS.get(self.task_type, {}).get(model_name, {})
        if not param_grid:
            return model, {}
        gs = GridSearchCV(model, param_grid, cv=cv, scoring=self.scoring,
                           n_jobs=-1, refit=True)
        gs.fit(X, y)
        result = {"best_params": gs.best_params_, "best_score": round(gs.best_score_, 4),
                   "method": "grid_search"}
        self.results[model_name] = result
        return gs.best_estimator_, result

    def random_search(self, model, model_name: str,
                       X: np.ndarray, y: np.ndarray,
                       n_iter: int = 20, cv: int = 3) -> Tuple[Any, Dict]:
        param_grid = PARAM_GRIDS.get(self.task_type, {}).get(model_name, {})
        if not param_grid:
            return model, {}
        rs = RandomizedSearchCV(model, param_grid, n_iter=min(n_iter, 10),
                                 cv=cv, scoring=self.scoring,
                                 n_jobs=-1, random_state=self.random_state, refit=True)
        rs.fit(X, y)
        result = {"best_params": rs.best_params_, "best_score": round(rs.best_score_, 4),
                   "method": "random_search"}
        self.results[model_name] = result
        return rs.best_estimator_, result

    def bayesian_optimization(self, model_class, model_name: str,
                               X: np.ndarray, y: np.ndarray,
                               n_trials: int = 30, cv: int = 3) -> Tuple[Any, Dict]:
        param_fn = OPTUNA_PARAMS.get(self.task_type, {}).get(model_name)
        if not param_fn:
            return model_class(), {}

        def objective(trial):
            params = param_fn(trial)
            model = model_class(**params)
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring)
                return scores.mean()
            except Exception:
                return float("-inf")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_model = model_class(**best_params)
        best_model.fit(X, y)
        result = {"best_params": best_params,
                   "best_score": round(study.best_value, 4),
                   "method": "bayesian_optuna",
                   "n_trials": n_trials}
        self.results[model_name] = result
        return best_model, result

    def optimize_best_model(self, model, model_name: str,
                             X: np.ndarray, y: np.ndarray,
                             method: str = "random") -> Tuple[Any, Dict]:
        """Optimize a model with selected method."""
        if method == "grid":
            return self.grid_search(model, model_name, X, y)
        elif method == "bayesian":
            model_class = type(model)
            return self.bayesian_optimization(model_class, model_name, X, y)
        else:
            return self.random_search(model, model_name, X, y)

    def get_results(self) -> Dict[str, Any]:
        return self.results

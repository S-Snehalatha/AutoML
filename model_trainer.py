"""
Module 4: AutoModelTrainer
Trains multiple algorithms with hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical as BayesCategorical
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False


class AutoModelTrainer:
    def __init__(self, task_type="Classification", max_models=8,
                 hp_method="Random Search", cv_folds=5, feature_names=None):
        self.task_type = task_type
        self.max_models = max_models
        self.hp_method = hp_method
        self.cv_folds = cv_folds
        self.feature_names = feature_names

    def _get_model_configs(self):
        if self.task_type == "Classification":
            configs = [
                ("Logistic Regression",
                 LogisticRegression(max_iter=1000, random_state=42),
                 {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}),
                ("Decision Tree",
                 DecisionTreeClassifier(random_state=42),
                 {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}),
                ("Random Forest",
                 RandomForestClassifier(n_estimators=100, random_state=42),
                 {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10, None]}),
                ("Gradient Boosting",
                 GradientBoostingClassifier(random_state=42),
                 {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5]}),
                ("SVM",
                 SVC(probability=True, random_state=42),
                 {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}),
                ("KNN",
                 KNeighborsClassifier(),
                 {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}),
                ("Neural Network",
                 MLPClassifier(max_iter=500, random_state=42),
                 {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "alpha": [0.0001, 0.001]}),
            ]
            if HAS_XGB:
                configs.append(("XGBoost",
                    XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
                    {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}))
            if HAS_LGBM:
                configs.append(("LightGBM",
                    LGBMClassifier(random_state=42, verbose=-1),
                    {"n_estimators": [50, 100], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}))
            if HAS_CB:
                configs.append(("CatBoost",
                    CatBoostClassifier(random_state=42, verbose=0),
                    {"iterations": [50, 100], "depth": [3, 5], "learning_rate": [0.05, 0.1]}))

        elif self.task_type == "Regression":
            configs = [
                ("Linear Regression", LinearRegression(), {}),
                ("Ridge", Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]}),
                ("Lasso", Lasso(max_iter=5000), {"alpha": [0.001, 0.01, 0.1, 1.0]}),
                ("Decision Tree",
                 DecisionTreeRegressor(random_state=42),
                 {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}),
                ("Random Forest",
                 RandomForestRegressor(n_estimators=100, random_state=42),
                 {"n_estimators": [50, 100], "max_depth": [3, 5, 10, None]}),
                ("Gradient Boosting",
                 GradientBoostingRegressor(random_state=42),
                 {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5]}),
                ("SVR", SVR(), {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}),
                ("KNN", KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 11]}),
                ("Neural Network",
                 MLPRegressor(max_iter=500, random_state=42),
                 {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "alpha": [0.0001, 0.001]}),
            ]
            if HAS_XGB:
                configs.append(("XGBoost",
                    XGBRegressor(random_state=42, verbosity=0),
                    {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}))
            if HAS_LGBM:
                configs.append(("LightGBM",
                    LGBMRegressor(random_state=42, verbose=-1),
                    {"n_estimators": [50, 100], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}))
            if HAS_CB:
                configs.append(("CatBoost",
                    CatBoostRegressor(random_state=42, verbose=0),
                    {"iterations": [50, 100], "depth": [3, 5], "learning_rate": [0.05, 0.1]}))

        else:  # Clustering
            configs = [
                ("K-Means (k=2)", KMeans(n_clusters=2, random_state=42, n_init=10), {}),
                ("K-Means (k=3)", KMeans(n_clusters=3, random_state=42, n_init=10), {}),
                ("K-Means (k=4)", KMeans(n_clusters=4, random_state=42, n_init=10), {}),
                ("K-Means (k=5)", KMeans(n_clusters=5, random_state=42, n_init=10), {}),
                ("DBSCAN", DBSCAN(eps=0.5, min_samples=5), {}),
                ("Hierarchical (k=3)", AgglomerativeClustering(n_clusters=3), {}),
                ("Hierarchical (k=5)", AgglomerativeClustering(n_clusters=5), {}),
            ]

        return configs[:self.max_models]

    def train(self, X, y, progress_callback=None):
        configs = self._get_model_configs()
        trained_models = []
        leaderboard_rows = []

        for i, (name, model, param_grid) in enumerate(configs):
            try:
                fitted_model, metrics = self._train_single(model, param_grid, X, y)
                trained_models.append({"name": name, "model": fitted_model, "metrics": metrics})
                row = {"Model": name}
                row.update(metrics)
                leaderboard_rows.append(row)
                if progress_callback:
                    progress_callback(name, i, len(configs), metrics)
            except Exception as e:
                print(f"Failed to train {name}: {e}")
                if progress_callback:
                    progress_callback(f"{name} (failed)", i, len(configs), {})

        return trained_models, leaderboard_rows

    def _train_single(self, model, param_grid, X, y):
        if self.task_type == "Clustering":
            return self._train_clustering(model, X)

        if not param_grid:
            model.fit(X, y)
            return model, self._compute_cv_metrics(model, X, y)

        cv = min(self.cv_folds, 3)
        scoring = "f1_weighted" if self.task_type == "Classification" else "r2"

        if self.hp_method == "Grid Search":
            searcher = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        elif self.hp_method == "Bayesian Optimization" and HAS_SKOPT:
            bayes_params = {k: BayesCategorical(v) if isinstance(v, list) else v
                            for k, v in param_grid.items()}
            searcher = BayesSearchCV(model, bayes_params, n_iter=12, cv=cv,
                                     scoring=scoring, n_jobs=-1, random_state=42, refit=True)
        else:
            searcher = RandomizedSearchCV(model, param_grid, n_iter=8, cv=cv,
                                          scoring=scoring, n_jobs=-1, random_state=42, refit=True)

        searcher.fit(X, y)
        best = searcher.best_estimator_
        return best, self._compute_cv_metrics(best, X, y)

    def _train_clustering(self, model, X):
        labels = model.fit_predict(X)
        unique_labels = np.unique(labels)
        valid = unique_labels[unique_labels != -1]
        score = 0.0
        if len(valid) >= 2:
            try:
                score = silhouette_score(X, labels)
            except Exception:
                pass
        model.labels_ = labels
        return model, {"Silhouette Score": round(float(score), 4), "Clusters": int(len(valid))}

    def _compute_cv_metrics(self, model, X, y):
        cv = min(self.cv_folds, 5)
        if self.task_type == "Classification":
            try:
                acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
            except Exception:
                model.fit(X, y)
                acc = accuracy_score(y, model.predict(X))
            try:
                f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted").mean()
            except Exception:
                f1 = 0.0
            return {"Accuracy": round(float(acc), 4), "F1 Score": round(float(f1), 4)}
        else:
            try:
                r2 = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
            except Exception:
                model.fit(X, y)
                r2 = r2_score(y, model.predict(X))
            try:
                from sklearn.metrics import mean_squared_error
                rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv,
                                                scoring="neg_mean_squared_error").mean())
            except Exception:
                rmse = 0.0
            return {"R² Score": round(float(r2), 4), "RMSE": round(float(rmse), 4)}

"""
Module 4: Model Training
Trains all ML algorithms based on task type
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               AdaBoostClassifier, AdaBoostRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, Any, List, Optional, Tuple
import time


class ModelTrainer:
    """Trains multiple models and tracks performance."""

    def __init__(self, task_type: str, random_state: int = 42):
        self.task_type = task_type
        self.random_state = random_state
        self.trained_models: Dict[str, Any] = {}
        self.training_times: Dict[str, float] = {}

    def get_model_configs(self) -> Dict[str, Any]:
        """Return all model configs for the task type."""
        if self.task_type == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=500, random_state=self.random_state),
                "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
                "SVM": SVC(probability=True, random_state=self.random_state),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Naive Bayes": GaussianNB(),
                "XGBoost": xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False,
                                              eval_metric="logloss", verbosity=0),
                "LightGBM": lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
                "Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                                                 random_state=self.random_state),
            }
            if CATBOOST_AVAILABLE:
                models["CatBoost"] = CatBoostClassifier(verbose=0, random_seed=self.random_state)
        elif self.task_type == "regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(random_state=self.random_state),
                "Lasso Regression": Lasso(random_state=self.random_state),
                "Decision Tree": DecisionTreeRegressor(random_state=self.random_state),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(random_state=self.random_state),
                "SVM": SVR(),
                "KNN": KNeighborsRegressor(n_neighbors=5),
                "XGBoost": xgb.XGBRegressor(random_state=self.random_state, verbosity=0),
                "LightGBM": lgb.LGBMRegressor(random_state=self.random_state, verbose=-1),
                "Neural Network": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200,
                                                random_state=self.random_state),
            }
            if CATBOOST_AVAILABLE:
                models["CatBoost"] = CatBoostRegressor(verbose=0, random_seed=self.random_state)
        elif self.task_type == "clustering":
            models = {
                "KMeans": KMeans(n_clusters=3, random_state=self.random_state, n_init=10),
                "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
                "Hierarchical": AgglomerativeClustering(n_clusters=3),
            }
        else:
            models = {}
        return models

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                   models_to_skip: List[str] = None) -> Dict[str, Any]:
        """Train all models and return fitted instances."""
        model_configs = self.get_model_configs()
        if models_to_skip:
            model_configs = {k: v for k, v in model_configs.items() if k not in models_to_skip}

        for name, model in model_configs.items():
            try:
                start = time.time()
                if self.task_type == "clustering":
                    model.fit(X_train)
                else:
                    model.fit(X_train, y_train)
                elapsed = round(time.time() - start, 3)
                self.trained_models[name] = model
                self.training_times[name] = elapsed
                print(f"  ✓ {name} trained in {elapsed}s")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")

        return self.trained_models

    def train_single(self, name: str, model, X_train: np.ndarray,
                      y_train: np.ndarray) -> Any:
        """Train a single model."""
        start = time.time()
        if self.task_type == "clustering":
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)
        self.trained_models[name] = model
        self.training_times[name] = round(time.time() - start, 3)
        return model

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained.")
        model = self.trained_models[model_name]
        return model.predict(X)

    def predict_proba(self, model_name: str, X: np.ndarray) -> Optional[np.ndarray]:
        model = self.trained_models.get(model_name)
        if model and hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return None

    def get_model(self, name: str):
        return self.trained_models.get(name)

    def get_all_models(self) -> Dict[str, Any]:
        return self.trained_models

    def get_training_times(self) -> Dict[str, float]:
        return self.training_times

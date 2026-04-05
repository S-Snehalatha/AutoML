"""
Module 3: Automated Preprocessing
Handles imputation, encoding, scaling, outlier removal, feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, f_classif, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
warnings.filterwarnings("ignore")


class AutoPreprocessor:
    """Full automated preprocessing pipeline."""

    def __init__(self, num_impute="median", cat_impute="most_frequent",
                 cat_encode="auto", scaler="standard", outlier_method="iqr",
                 feat_select="auto", target_col=None, task_type="Classification"):
        self.num_impute = num_impute
        self.cat_impute = cat_impute
        self.cat_encode = cat_encode
        self.scaler = scaler
        self.outlier_method = outlier_method
        self.feat_select = feat_select
        self.target_col = target_col
        self.task_type = task_type

        self.num_cols_ = []
        self.cat_cols_ = []
        self.num_imputer_ = None
        self.cat_imputer_ = None
        self.encoders_ = {}
        self.scaler_ = None
        self.selector_ = None
        self.selected_features_ = None
        self.label_encoder_ = None
        self.feature_names_out_ = []

    def fit_transform(self, df: pd.DataFrame):
        report = {}
        df = df.copy()

        # Drop completely empty columns
        df.dropna(axis=1, how="all", inplace=True)
        df.drop_duplicates(inplace=True)
        report["input_features"] = len(df.columns) - (1 if self.target_col else 0)

        # Split features / target
        if self.target_col and self.target_col in df.columns:
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col].copy()
        else:
            X = df.copy()
            y = None

        # Identify column types
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # ── Outlier removal (before imputation) ──────────────
        rows_before = len(X)
        if self.outlier_method != "none" and self.num_cols_:
            X, y = self._remove_outliers(X, y)
        report["outliers_removed"] = rows_before - len(X)
        report["rows_after"] = len(X)

        # ── Numeric imputation ───────────────────────────────
        if self.num_cols_:
            X = self._impute_numeric(X)

        # ── Categorical imputation ───────────────────────────
        if self.cat_cols_:
            X = self._impute_categorical(X)

        # ── Categorical encoding ─────────────────────────────
        if self.cat_cols_:
            X = self._encode_categorical(X)

        # ── Scaling ──────────────────────────────────────────
        if self.scaler != "none" and self.num_cols_:
            X = self._scale(X)

        # ── Feature selection ────────────────────────────────
        feature_names = list(X.columns)
        if self.feat_select != "none" and y is not None and self.task_type != "Clustering":
            X, feature_names = self._select_features(X, y, feature_names)

        self.feature_names_out_ = feature_names
        report["output_features"] = len(feature_names)

        # ── Encode target ────────────────────────────────────
        if y is not None and self.task_type == "Classification" and y.dtype == object:
            self.label_encoder_ = LabelEncoder()
            y_enc = self.label_encoder_.fit_transform(y)
        elif y is not None:
            y_enc = y.values if hasattr(y, "values") else y
        else:
            y_enc = None

        X_arr = X.values if hasattr(X, "values") else X
        return X_arr, y_enc, feature_names, report

    # ── internals ─────────────────────────────────────────────

    def _remove_outliers(self, X, y):
        if self.outlier_method == "iqr":
            mask = np.ones(len(X), dtype=bool)
            for col in self.num_cols_:
                Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                IQR = Q3 - Q1
                mask &= (X[col] >= Q1 - 3 * IQR) & (X[col] <= Q3 + 3 * IQR)
            X = X[mask]
            if y is not None:
                y = y[mask]

        elif self.outlier_method == "zscore":
            from scipy import stats
            mask = np.all(np.abs(stats.zscore(X[self.num_cols_].fillna(0))) < 3.5, axis=1)
            X = X[mask]
            if y is not None:
                y = y[mask]

        elif self.outlier_method == "isolation_forest":
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(X[self.num_cols_].fillna(0))
            mask = preds == 1
            X = X[mask]
            if y is not None:
                y = y[mask]

        return X.reset_index(drop=True), (y.reset_index(drop=True) if y is not None and hasattr(y, "reset_index") else y)

    def _impute_numeric(self, X):
        if self.num_impute == "knn":
            self.num_imputer_ = KNNImputer(n_neighbors=5)
        elif self.num_impute == "iterative":
            self.num_imputer_ = IterativeImputer(random_state=42)
        else:
            self.num_imputer_ = SimpleImputer(strategy=self.num_impute)
        X[self.num_cols_] = self.num_imputer_.fit_transform(X[self.num_cols_])
        return X

    def _impute_categorical(self, X):
        self.cat_imputer_ = SimpleImputer(strategy=self.cat_impute, fill_value="missing")
        imputed=self.cat_imputer_.fit_transform(X[self.cat_cols_])
        X[self.cat_cols_] = pd.DataFrame(
            imputed,
            columns=self.cat_cols_,
            index=X.index
        )
        return X

    def _encode_categorical(self, X):
        strategy = self.cat_encode
        for col in self.cat_cols_:
            n_unique = X[col].nunique()
            use_onehot = (strategy == "onehot") or (strategy == "auto" and n_unique <= 10)
            if use_onehot:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            else:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X[col] = enc.fit_transform(X[[col]])
                self.encoders_[col] = enc
        return X

    def _scale(self, X):
        num_present = [c for c in self.num_cols_ if c in X.columns]
        if not num_present:
            return X
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        self.scaler_ = scalers.get(self.scaler, StandardScaler())
        X[num_present] = self.scaler_.fit_transform(X[num_present])
        return X

    def _select_features(self, X, y, feature_names):
        k = min(max(5, len(feature_names) // 2), len(feature_names))
        try:
            if self.task_type == "Regression":
                score_fn = f_regression
            else:
                score_fn = f_classif

            selector = SelectKBest(score_func=score_fn, k=k)
            X_arr = X.values if hasattr(X, "values") else X
            X_new = selector.fit_transform(X_arr, y)
            selected = selector.get_support()
            selected_names = [feature_names[i] for i, s in enumerate(selected) if s]
            self.selector_ = selector
            return pd.DataFrame(X_new, columns=selected_names), selected_names
        except Exception:
            return X, feature_names

    def transform(self, df: pd.DataFrame):
        """Transform new data using fitted pipeline."""
        df = df.copy()
        if self.target_col and self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])

        if self.num_cols_:
            available_num = [c for c in self.num_cols_ if c in df.columns]
            if available_num and self.num_imputer_:
                df[available_num] = self.num_imputer_.transform(df[available_num])

        if self.cat_cols_:
            available_cat = [c for c in self.cat_cols_ if c in df.columns]
            if available_cat and self.cat_imputer_:
                df[available_cat] = self.cat_imputer_.transform(df[available_cat])

        for col, enc in self.encoders_.items():
            if col in df.columns:
                df[col] = enc.transform(df[[col]])

        # One-hot encode
        for col in self.cat_cols_:
            if col in df.columns and col not in self.encoders_:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        if self.scaler_:
            num_present = [c for c in self.num_cols_ if c in df.columns]
            if num_present:
                df[num_present] = self.scaler_.transform(df[num_present])

        if self.selector_:
            try:
                arr = df.values if hasattr(df, "values") else df
                arr = self.selector_.transform(arr)
                return arr
            except Exception:
                pass

        # Align columns
        try:
            for col in self.feature_names_out_:
                if col not in df.columns:
                    df[col] = 0
            return df[self.feature_names_out_].values
        except Exception:
            return df.values

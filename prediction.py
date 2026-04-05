"""
Module: PredictionEngine
Robust single + batch prediction with full preprocessing replay.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class PredictionEngine:
    def __init__(self, model, preprocessor, task_type, feature_names):
        self.model        = model
        self.preprocessor = preprocessor
        self.task_type    = task_type
        self.feature_names = feature_names  # final feature names after preprocessing

    # ── public API ────────────────────────────────────────────

    def predict(self, input_df: pd.DataFrame):
        """Single-row prediction. Returns (prediction, confidence)."""
        X = self._safe_transform(input_df)
        raw = self.model.predict(X)
        pred = self._decode_label(raw[0])
        confidence = self._get_confidence(X)
        return pred, confidence

    def batch_predict(self, df: pd.DataFrame):
        """Multi-row prediction. Returns array of predictions."""
        X    = self._safe_transform(df)
        raws = self.model.predict(X)
        return self._decode_labels(raws)

    # ── internals ─────────────────────────────────────────────

    def _safe_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform input dataframe through the fitted preprocessor.
        Handles column alignment, missing columns, and one-hot drift.
        """
        df = df.copy()

        # Drop target column if accidentally included
        tgt = getattr(self.preprocessor, "target_col", None)
        if tgt and tgt in df.columns:
            df = df.drop(columns=[tgt])

        try:
            X = self.preprocessor.transform(df)

            # Ensure correct shape — align to trained feature count
            if hasattr(X, "shape") and self.feature_names:
                expected = len(self.feature_names)
                if X.shape[1] != expected:
                    X = self._align_array(X, df)
            return X

        except Exception as e:
            # Fallback: manual alignment to feature_names
            return self._manual_align(df)

    def _manual_align(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build the feature array by manually matching feature_names.
        Handles one-hot columns that may not appear in single-row input.
        """
        if not self.feature_names:
            return df.select_dtypes(include=[np.number]).values

        # Apply basic numeric imputation + one-hot manually
        proc = self.preprocessor

        # Step 1: numeric imputation
        num_cols = getattr(proc, "num_cols_", [])
        for col in num_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median() if not df[col].dropna().empty else 0)

        # Step 2: categorical imputation
        cat_cols = getattr(proc, "cat_cols_", [])
        for col in cat_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna("missing")

        # Step 3: ordinal encoding for high-cardinality categoricals
        encoders = getattr(proc, "encoders_", {})
        for col, enc in encoders.items():
            if col in df.columns:
                try:
                    df[col] = enc.transform(df[[col]])
                except Exception:
                    df[col] = 0

        # Step 4: one-hot encode remaining categoricals
        for col in cat_cols:
            if col in df.columns and col not in encoders:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        # Step 5: scaling
        scaler = getattr(proc, "scaler_", None)
        if scaler:
            num_present = [c for c in num_cols if c in df.columns]
            if num_present:
                try:
                    df[num_present] = scaler.transform(df[num_present])
                except Exception:
                    pass

        # Step 6: align to exact feature_names list
        result = {}
        for feat in self.feature_names:
            if feat in df.columns:
                result[feat] = df[feat].values[0] if len(df) == 1 else df[feat].values
            else:
                result[feat] = 0  # missing one-hot column → 0

        aligned = pd.DataFrame([result] if len(df) == 1 else result)
        return aligned[self.feature_names].values

    def _align_array(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Trim or pad array to match expected feature count."""
        expected = len(self.feature_names)
        if X.shape[1] == expected:
            return X
        elif X.shape[1] > expected:
            return X[:, :expected]
        else:
            # Pad with zeros
            pad = np.zeros((X.shape[0], expected - X.shape[1]))
            return np.hstack([X, pad])

    def _decode_label(self, raw_pred):
        """Convert numeric label back to original class name."""
        le = getattr(self.preprocessor, "label_encoder_", None)
        if self.task_type == "Classification" and le is not None:
            try:
                val = int(raw_pred)
                return le.inverse_transform([val])[0]
            except Exception:
                return raw_pred
        elif self.task_type == "Regression":
            try:
                return round(float(raw_pred), 4)
            except Exception:
                return raw_pred
        else:
            try:
                return int(raw_pred)
            except Exception:
                return raw_pred

    def _decode_labels(self, raw_preds):
        """Batch label decoding."""
        le = getattr(self.preprocessor, "label_encoder_", None)
        if self.task_type == "Classification" and le is not None:
            try:
                return le.inverse_transform(raw_preds.astype(int))
            except Exception:
                pass
        if self.task_type == "Regression":
            return np.round(raw_preds.astype(float), 4)
        return raw_preds

    def _get_confidence(self, X: np.ndarray):
        """Return max probability for classification, None otherwise."""
        if self.task_type != "Classification":
            return None
        if not hasattr(self.model, "predict_proba"):
            return None
        try:
            proba = self.model.predict_proba(X)[0]
            return float(max(proba))
        except Exception:
            return None

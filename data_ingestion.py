"""
Module 1: Data Ingestion
Handles CSV, Excel, JSON loading with schema detection.
"""

import pandas as pd
import numpy as np
import io
import json


class DataIngestion:
    """Loads any tabular dataset and detects its schema."""

    def load(self, uploaded_file):
        name = uploaded_file.name.lower()
        try:
            if name.endswith(".csv"):
                df = self._load_csv(uploaded_file)
            elif name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            elif name.endswith(".json"):
                df = self._load_json(uploaded_file)
            else:
                return None, None

            df = self._clean_column_names(df)
            schema = self._detect_schema(df)
            return df, schema
        except Exception as e:
            import streamlit as st
            st.error(f"Failed to load file: {e}")
            return None, None

    def _load_csv(self, f):
        content = f.read()
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(io.BytesIO(content), encoding=enc)
            except Exception:
                continue
        raise ValueError("Cannot decode CSV file")

    def _load_json(self, f):
        content = f.read()
        data = json.loads(content)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if "data" in data:
                return pd.DataFrame(data["data"])
            return pd.DataFrame([data])
        return pd.DataFrame(data)

    def _clean_column_names(self, df):
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(r"[^\w\s]", "_", regex=True)
            .str.replace(r"\s+", "_", regex=True)
            .str.lower()
        )
        return df

    def _detect_schema(self, df):
        column_types = {}
        missing_pct = {}
        n_unique = {}

        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                col_type = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = "datetime"
            else:
                col_type = "categorical"

            column_types[col] = col_type
            missing_pct[col] = df[col].isna().mean() * 100
            n_unique[col] = int(df[col].nunique())

        numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
        cat_cols = [c for c, t in column_types.items() if t == "categorical"]

        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_numeric": len(numeric_cols),
            "n_categorical": len(cat_cols),
            "column_types": column_types,
            "missing_pct": missing_pct,
            "n_unique": n_unique,
            "numeric_cols": numeric_cols,
            "categorical_cols": cat_cols,
            "total_missing_pct": df.isna().mean().mean() * 100,
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        }

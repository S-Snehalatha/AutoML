"""Module 9: NLP Output Generator - human-readable predictions."""

import random


class NLPOutputGenerator:
    """Converts model predictions into natural language explanations."""

    def generate(self, prediction, confidence, task_type, target_col, input_data):
        target_col = target_col or "target"
        col_display = target_col.replace("_", " ").title()

        if task_type == "Regression":
            return self._regression_nl(prediction, col_display, input_data)
        elif task_type == "Classification":
            return self._classification_nl(prediction, confidence, col_display, input_data)
        else:
            return self._clustering_nl(prediction, input_data)

    def _regression_nl(self, prediction, col, input_data):
        # Format number nicely
        if abs(prediction) > 1000:
            formatted = f"{prediction:,.2f}"
        elif abs(prediction) > 1:
            formatted = f"{prediction:.4f}"
        else:
            formatted = f"{prediction:.6f}"

        # Currency detection
        currency_cols = ["price", "cost", "salary", "income", "revenue", "sales", "amount", "fee", "wage"]
        is_currency = any(kw in col.lower() for kw in currency_cols)

        if is_currency:
            display = f"${formatted}" if "." in formatted else f"${formatted}"
        else:
            display = formatted

        top_features = list(input_data.keys())[:3]
        feature_str = ", ".join(f"**{f.replace('_',' ')}** = {v}" for f, v in list(input_data.items())[:3])

        templates = [
            f"Based on the provided data ({feature_str}), the predicted **{col}** is **{display}**.",
            f"Using the input features, the model estimates the **{col}** to be approximately **{display}**.",
            f"The AutoML model predicts a **{col}** of **{display}** based on the given characteristics.",
        ]
        return random.choice(templates)

    def _classification_nl(self, prediction, confidence, col, input_data):
        conf_str = f" (confidence: {confidence:.1%})" if confidence else ""
        feature_str = ", ".join(f"**{k.replace('_',' ')}** = {v}" for k, v in list(input_data.items())[:3])

        templates = [
            f"Based on the input features ({feature_str}), the model predicts the **{col}** to be **{prediction}**{conf_str}.",
            f"With the provided characteristics, the predicted **{col}** is **{prediction}**{conf_str}.",
            f"The classification model assigns this record to the class **{prediction}**{conf_str} based on its features.",
        ]
        return random.choice(templates)

    def _clustering_nl(self, prediction, input_data):
        return (f"This data point has been assigned to **Cluster {prediction}** by the clustering model. "
                f"Records in this cluster share similar characteristics and patterns.")

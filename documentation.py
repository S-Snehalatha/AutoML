"""Module 11: Documentation Generator."""

import streamlit as st


class DocumentationGenerator:
    def render_in_streamlit(self, session_state=None):
        st.markdown("""
## 📚 AutoML Studio — User Manual

---

### 1. Getting Started

**AutoML Studio** automates the full machine learning pipeline from raw data to deployed predictions.
No coding required. Simply upload your dataset and let the system handle the rest.

---

### 2. Data Upload

**Supported Formats:** CSV, Excel (.xlsx/.xls), JSON

**Steps:**
1. Click **"Browse files"** in the Data Upload tab
2. Select your dataset file
3. The system automatically detects column types (numeric, categorical, datetime)
4. Choose your **Target Column** (the variable you want to predict)
5. Confirm or override the detected **Task Type** (Classification, Regression, or Clustering)

**Tips:**
- Ensure your CSV uses standard delimiters (comma or semicolon)
- Column names should be in the first row
- The system handles most encodings automatically (UTF-8, Latin-1)

---

### 3. Exploratory Data Analysis (EDA)

Click **"Run Full EDA"** to automatically generate:

| Visualization | Description |
|---|---|
| Histograms | Distribution of each numeric feature |
| Box Plots | Outlier detection and spread visualization |
| Correlation Heatmap | Pairwise linear relationships between features |
| Scatter Plots | Relationship between features and the target |
| Bar / Count Plots | Frequency of categorical values |
| Missing Value Heatmap | Visual identification of null/missing data |
| Pair Plot | All pairwise feature relationships |

---

### 4. Preprocessing

The system automatically handles:

| Step | Methods Available |
|---|---|
| **Missing Values** | Median, Mean, KNN Imputation, Iterative (MICE) |
| **Categorical Encoding** | One-Hot, Label/Ordinal, Auto (based on cardinality) |
| **Feature Scaling** | StandardScaler, MinMaxScaler, RobustScaler |
| **Outlier Removal** | IQR method, Z-Score, Isolation Forest |
| **Feature Selection** | SelectKBest (F-test), Mutual Information, RFE |

---

### 5. Model Training

AutoML Studio trains up to **14 algorithms** simultaneously:

**Classification:**
Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Neural Network, XGBoost, LightGBM, CatBoost

**Regression:**
Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN, Neural Network, XGBoost, LightGBM, CatBoost

**Clustering:**
K-Means (k=2 to 5), DBSCAN, Hierarchical Clustering

**Hyperparameter Optimization:**
- **Random Search**: Fast, samples random combinations (default)
- **Grid Search**: Exhaustive search over parameter grid
- **Bayesian Optimization**: Smart sequential search (requires scikit-optimize)

---

### 6. Model Evaluation

#### Classification Metrics
| Metric | Description |
|---|---|
| Accuracy | % correctly classified |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | Harmonic mean of Precision & Recall |
| ROC AUC | Area under the ROC curve (0.5 = random, 1.0 = perfect) |

#### Regression Metrics
| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² Score | Proportion of variance explained (1.0 = perfect) |

#### Visual Analyses
- **Confusion Matrix** — True vs predicted class distribution
- **ROC Curve** — True Positive vs False Positive rate trade-off
- **Precision-Recall Curve** — Precision vs Recall trade-off
- **Predicted vs Actual** — Visual accuracy for regression
- **Residual Plot** — Model error patterns
- **Learning Curves** — Training vs validation performance over data size
- **Lift & Cumulative Gain Charts** — Model ranking performance

---

### 7. Explainability (SHAP)

**SHAP (SHapley Additive exPlanations)** explains individual predictions:

| Plot | Interpretation |
|---|---|
| Feature Importance | Which features matter most on average |
| SHAP Summary | Feature impact distribution across all predictions |
| SHAP Beeswarm | Individual contribution of each data point |
| SHAP Dependence | How a single feature affects predictions |

> **Note:** SHAP requires the `shap` package. For complex models, SHAP computation may take 1-2 minutes.

---

### 8. Making Predictions

**Manual Input:**
1. Go to the **Prediction** tab
2. Enter values for each feature using the input form
3. Click **"Predict"**
4. View the prediction with confidence score and natural language explanation

**Batch Prediction:**
1. Upload a CSV file with the same features as your training data (without the target column)
2. Click **"Batch Predict"**
3. Download the results CSV with predictions appended

---

### 9. Best Model Selection

The system automatically selects the best model based on:
- **Classification**: Highest F1 Score (weighted)
- **Regression**: Highest R² Score
- **Clustering**: Highest Silhouette Score

---

### 10. API Deployment

The FastAPI backend (`/backend/api/`) exposes these endpoints:

```
POST /predict          — Single prediction
POST /batch_predict    — Batch predictions from CSV
GET  /model/info       — Model metadata and metrics
GET  /health           — Health check
```

**Example API call:**
```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"feature1": 1.5, "feature2": "category_A"}'
```

---

### 11. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Streamlit UI:  http://localhost:8501
# FastAPI docs:  http://localhost:8000/docs
```

---

### 12. Interpreting Results

**For Classification:**
- A ROC AUC > 0.85 indicates a strong model
- F1 Score is more informative than Accuracy for imbalanced datasets
- Check the Confusion Matrix to understand error types

**For Regression:**
- R² > 0.8 indicates the model explains 80%+ of variance
- Check Residual Plots — patterns indicate model bias
- RMSE should be interpreted relative to your target variable's scale

**For Clustering:**
- Silhouette Score > 0.5 indicates well-separated clusters
- Use Cluster Plots to visually inspect cluster quality
- DBSCAN is better for irregular shapes; K-Means for spherical clusters

---

### 13. Troubleshooting

| Issue | Solution |
|---|---|
| "Failed to load file" | Check encoding (try saving as UTF-8 CSV) |
| Model training very slow | Reduce max_models or use Random Search |
| SHAP not available | Install: `pip install shap` |
| Low accuracy | Try different preprocessing options or more data |
| Memory error | Sample your dataset to < 100K rows |

---

*AutoML Studio — Built with ❤️ using scikit-learn, XGBoost, LightGBM, CatBoost, SHAP & Streamlit*
        """)

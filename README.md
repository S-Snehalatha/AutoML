# ⚡ AutoML Studio

Enterprise-grade Automated Machine Learning pipeline — similar to DataRobot and H2O AutoML.

## 🚀 Quick Start

### Option 1: Local Python
```bash
pip install -r requirements.txt
python run.py
```

### Option 2: Docker Compose
```bash
docker-compose up --build
```

Access:
- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs

---

## 📁 Project Structure

```
automl_app/
├── frontend/
│   └── app.py                    # Streamlit multi-tab UI
├── backend/
│   ├── modules/
│   │   ├── data_ingestion.py     # CSV/Excel/JSON loader + schema detection
│   │   ├── eda.py                # Auto EDA: histograms, heatmaps, scatter, etc.
│   │   ├── preprocessing.py      # Imputation, encoding, scaling, feature selection
│   │   ├── model_trainer.py      # 14 algorithms + Grid/Random/Bayesian HPO
│   │   ├── model_evaluator.py    # Full metrics + ROC, Confusion Matrix, etc.
│   │   ├── explainability.py     # SHAP values + feature importance
│   │   ├── model_selector.py     # Auto best-model selection
│   │   ├── prediction.py         # Single + batch prediction engine
│   │   ├── nlp_output.py         # Human-readable prediction explanations
│   │   ├── dashboard.py          # Analytics dashboard
│   │   └── documentation.py     # Built-in user manual
│   └── api/
│       └── main.py               # FastAPI REST API
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py                        # Launcher (Streamlit + FastAPI)
└── README.md
```

---

## 🤖 Supported Algorithms

### Classification
| Model | Library |
|-------|---------|
| Logistic Regression | scikit-learn |
| Decision Tree | scikit-learn |
| Random Forest | scikit-learn |
| Gradient Boosting | scikit-learn |
| SVM | scikit-learn |
| KNN | scikit-learn |
| Neural Network (MLP) | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| CatBoost | catboost |

### Regression
| Model | Library |
|-------|---------|
| Linear Regression | scikit-learn |
| Ridge / Lasso | scikit-learn |
| Decision Tree | scikit-learn |
| Random Forest | scikit-learn |
| Gradient Boosting | scikit-learn |
| SVR | scikit-learn |
| KNN | scikit-learn |
| Neural Network (MLP) | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| CatBoost | catboost |

### Clustering
- K-Means (k=2,3,4,5)
- DBSCAN
- Hierarchical (k=3,5)

---

## ⚙️ Hyperparameter Optimization
- **Random Search** (default, fast)
- **Grid Search** (exhaustive)
- **Bayesian Optimization** (requires `scikit-optimize`)

---

## 📊 EDA Visualizations
- Histograms & Distribution Plots
- Box Plots
- Correlation Heatmap
- Scatter Plots
- Bar & Count Plots
- Missing Value Heatmap
- Pair Plot

---

## 📈 Evaluation Plots
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Residual Plot
- Predicted vs Actual
- Learning Curves
- Lift Chart
- Cumulative Gain Chart

---

## 💡 Explainability
- SHAP Feature Importance
- SHAP Beeswarm Plot
- SHAP Dependence Plot
- Built-in Feature Importance (tree models)

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List registered models |
| GET | `/model/info/{id}` | Model metadata |
| POST | `/predict` | Single prediction |
| POST | `/batch_predict` | Batch CSV prediction |

### Example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "income": 55000, "education": "Bachelor"}}'
```

---

## 🐳 Docker

```bash
# Build
docker-compose build

# Run
docker-compose up

# Stop
docker-compose down
```

---

## 📦 Dependencies
- `streamlit` — UI framework
- `scikit-learn` — Core ML algorithms
- `xgboost`, `lightgbm`, `catboost` — Boosting algorithms
- `shap` — Model explainability
- `plotly` — Interactive visualizations
- `fastapi` + `uvicorn` — REST API
- `pandas`, `numpy`, `scipy` — Data processing
- `scikit-optimize` — Bayesian HPO (optional)

---

*AutoML Studio v1.0 — Built with Python, scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, Plotly, Streamlit & FastAPI*

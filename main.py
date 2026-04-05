"""
FastAPI Backend — Model Deployment & Prediction API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import io
import pandas as pd
import numpy as np

app = FastAPI(
    title="AutoML Studio API",
    description="Automated ML pipeline deployment and prediction service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_store: Dict[str, Any] = {}


class PredictRequest(BaseModel):
    features: Dict[str, Any]
    model_id: Optional[str] = "best"


class PredictResponse(BaseModel):
    prediction: Any
    confidence: Optional[float]
    explanation: str
    model_name: str


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(model_store)}


@app.get("/models")
def list_models():
    return {
        "models": [
            {"id": k, "name": v.get("name"), "task_type": v.get("task_type"),
             "metrics": v.get("metrics", {})}
            for k, v in model_store.items()
        ]
    }


@app.get("/model/info/{model_id}")
def model_info(model_id: str = "best"):
    if model_id not in model_store:
        raise HTTPException(404, f"Model '{model_id}' not found")
    m = model_store[model_id]
    return {
        "name": m.get("name"),
        "task_type": m.get("task_type"),
        "metrics": m.get("metrics", {}),
        "feature_names": m.get("feature_names", []),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    mid = request.model_id or "best"
    if mid not in model_store:
        raise HTTPException(404, "Model not found. Please register a model first.")

    m = model_store[mid]
    model = m["model"]
    preprocessor = m.get("preprocessor")
    task_type = m.get("task_type", "Classification")
    input_df = pd.DataFrame([request.features])

    try:
        X = preprocessor.transform(input_df) if preprocessor else input_df.values
        pred = model.predict(X)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(max(proba))

        if isinstance(pred, (np.integer,)):
            pred = int(pred)
        elif isinstance(pred, (np.floating,)):
            pred = float(pred)

        col = (m.get("target_col") or "target").replace("_", " ").title()
        if task_type == "Regression":
            explanation = f"The predicted {col} is {pred:.4f}."
        elif task_type == "Classification":
            conf_str = f" (confidence: {confidence:.1%})" if confidence else ""
            explanation = f"The model predicts class '{pred}' for {col}{conf_str}."
        else:
            explanation = f"This record belongs to cluster {pred}."

        return PredictResponse(prediction=pred, confidence=confidence,
                               explanation=explanation, model_name=m.get("name", "Unknown"))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...), model_id: str = "best"):
    if model_id not in model_store:
        raise HTTPException(404, "Model not found")
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    m = model_store[model_id]
    model = m["model"]
    preprocessor = m.get("preprocessor")
    try:
        X = preprocessor.transform(df) if preprocessor else df.values
        preds = model.predict(X).tolist()
        return {"predictions": preds, "n_samples": len(preds), "model_name": m.get("name")}
    except Exception as e:
        raise HTTPException(500, f"Batch prediction failed: {str(e)}")

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict
import joblib
import os
import random

# Load artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "xgb_final_model.pkl"))
final_features = joblib.load(os.path.join(MODEL_DIR, "final_features.pkl"))
threshold = joblib.load(os.path.join(MODEL_DIR, "optimal_threshold.pkl"))

app = FastAPI(title="Financial Distress Prediction API")

# 🔹 Example data for Swagger
example_features = {f: round(random.random(), 4) for f in final_features}


class CompanyFeatures(BaseModel):
    features: Dict[str, float] = Field(
        ..., example=example_features
    )


@app.post("/predict")
def predict(data: CompanyFeatures):

    df = pd.DataFrame([data.features])

    # Ensure correct feature order
    df = df[final_features]
    df = df.astype(float)

    proba = model.predict_proba(df)[:, 1][0]
    prediction = int(proba >= threshold)

    return {
        "probability": round(float(proba), 4),
        "prediction": prediction
    }

@app.get("/")
def root():
    return {"message": "Financial Distress Prediction API is running!"}

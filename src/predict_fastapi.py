# src/predict_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

from src.predict_utils import predict_single, predict_batch, optimal_features

app = FastAPI(title="Financial Distress Predictor API")

class CompanyFeatures(BaseModel):
    features: Dict[str, float]

class BatchCompanies(BaseModel):
    companies: List[CompanyFeatures]

@app.post("/predict")
def predict(company: CompanyFeatures):
    return predict_single(company.features)

@app.post("/predict_batch")
def predict_companies(batch: BatchCompanies):
    return predict_batch([c.features for c in batch.companies])

# src/predict_utils.py
import joblib
import pandas as pd
from typing import Dict, List
from fastapi import HTTPException

# =========================
# Load artifacts (ONCE)
# =========================
model = joblib.load("models/financial_distress_model.pkl")
scaler = joblib.load("models/scaler.pkl")
optimal_features = joblib.load("models/optimal_features.pkl")


# =========================
# Helpers
# =========================
def transform_input(input_data: Dict[str, float]):
    missing = set(optimal_features) - set(input_data.keys())
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {sorted(missing)}"
        )

    df = pd.DataFrame([input_data])[optimal_features]
    return scaler.transform(df)


def transform_batch_input(batch_data: List[Dict[str, float]]):
    dfs = []
    for i, company in enumerate(batch_data):
        missing = set(optimal_features) - set(company.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Company {i} missing features: {sorted(missing)}"
            )
        dfs.append(pd.DataFrame([company]))

    df_all = pd.concat(dfs, ignore_index=True)[optimal_features]
    return scaler.transform(df_all)


# =========================
# Prediction logic
# =========================
def predict_single(features: Dict[str, float]):
    X_scaled = transform_input(features)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    return {
        "prediction": int(pred),
        "label": "Distressed" if pred == 1 else "Not Distressed",
        "probability_distressed": float(prob),
    }


def predict_batch(batch_features: List[Dict[str, float]]):
    X_scaled = transform_batch_input(batch_features)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    results = []
    for p, pr in zip(preds, probs):
        results.append({
            "prediction": int(p),
            "label": "Distressed" if p == 1 else "Not Distressed",
            "probability_distressed": float(pr),
        })

    return {"results": results}

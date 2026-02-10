# Financial Distress Prediction – End-to-End ML & MLOps Pipeline

## 📌 Problem Description

This project predicts corporate financial distress using financial indicators. From a business perspective, missing a distressed company is more costly than falsely flagging a healthy one, so the system prioritizes high recall on the minority (distressed) class using threshold tuning.

## 🧠 Solution Overview

An end-to-end ML + MLOps pipeline covering:
* Data preprocessing & feature engineering
* Model training and tuning
* Business-driven threshold optimization
* Experiment tracking & model registry
* Automated workflow orchestration
* Containerized model deployment as a web service

## 🏗️ Architecture (High Level)

```
Data (CSV)
   ↓
train.py
- Feature engineering
- Stratified split (train/val/test)
- XGBoost tuning
- Threshold optimization
- Artifact saving
   ↓
MLflow
- Experiment tracking
- Model registry
   ↓
Prefect
- Orchestrates training + logging
   ↓
FastAPI
- Online inference service
   ↓
Docker
- Containerized deployment
```

## 🔬 EDA & Modeling

* Target imbalance analyzed (~3–4% distressed)
* Correlation-based feature selection
* Models tried: baseline + XGBoost
* Hyperparameter tuning via `RandomizedSearchCV`
* Threshold tuning to achieve recall ≥ 0.6 on distressed companies

## 🚀 Deployment

* FastAPI used for real-time inference
* Model, features, and threshold loaded as artifacts
* Dockerized for portability and cloud readiness
* REST endpoint: `/predict`

## ⚙️ MLOps Components

* **Experiment Tracking:** MLflow
* **Model Registry:** MLflow (Production stage)
* **Workflow Orchestration:** Prefect
* **Model Serving:** FastAPI
* **Containerization:** Docker
* **Reproducibility:** Saved artifacts + requirements.txt

## 📂 Project Structure

```
├── artifacts/
├── env/
├── src/
│   ├── train.py
│   ├── predict_fastapi.py
│   ├── mlflow_integration.py
│   ├── prefect_flow.py
│   └── test_predict.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── Notebook_Company_Distress_Prediction.ipynb
├── requirements.txt
└── README.md
```

## ▶️ How to Run (Quick)

```bash
pip install -r requirements.txt
python src/train.py
python src/prefect_flow.py
uvicorn src.predict_fastapi:app --port 8000
```

## 🏷️ Project Type

End-to-End Machine Learning & MLOps Project

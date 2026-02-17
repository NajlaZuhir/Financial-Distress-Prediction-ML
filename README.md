# 🏦 Financial Distress Prediction
An end-to-end machine learning project that classifies financially distressed companies using imbalanced learning techniques.

## 📊 Project Overview
This project aims to identify financially distressed companies; those struggling to pay employees, meet bill payments, or fulfil other financial obligations. Early detection is critical, as the cost of missing a distressed firm far outweighs the cost of falsely flagging a healthy one. This asymmetry makes the problem inherently cost-sensitive, shifting the focus toward minimizing false negatives (catching as many distressed cases) and driving the need to explicitly address the class imbalance present in the data.

## 🔧 Tech Stack
- Python 3.10
- scikit-learn, xgboost, imbalanced-learn
- pandas, numpy
- FastAPI, Streamlit
- Docker

## 📊 Dataset
- **Source:** [Financial Distress Dataset](https://www.kaggle.com/datasets/shebrahimi/financial-distress) from Kaggle
- **Records:** 3,672
- **Features:** 86

## 📁 Project Structure
```
├── artifacts/         # Saved models & datasets
├── src/
│   ├── train.py       # Model training with undersampling
│   ├── predict.py     # Predictions & evaluation
├── main.py            # FastAPI app
├── app.py             # Streamlit UI
├── Dockerfile
└── requirements.txt
```

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python src/train.py

# Run API
uvicorn main:app --reload

# Run UI (separate terminal)
streamlit run app.py
```

**Machine Learning Project | 2026**

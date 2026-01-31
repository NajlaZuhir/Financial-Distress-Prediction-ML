import requests
import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load test set saved by train.py
test_df = pd.read_csv(os.path.join(MODEL_DIR, "test.csv"))

url = "http://127.0.0.1:8001/predict"

# Pick 5 random test samples
# samples = test_df.sample(5, random_state=42)
samples = test_df.sample(5)  # no random_state


for idx, row in samples.iterrows():
    features = row.drop("Financial Distress").to_dict()
    payload = {"features": features}

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(f"\nSample index: {idx}")
        print("Actual:", int(row["Financial Distress"]))
        print("Prediction:", response.json())
    else:
        print("Error:", response.status_code, response.text)

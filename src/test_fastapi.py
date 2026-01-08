import requests
import random
import joblib

# Load optimal features from your saved file
optimal_features = joblib.load("/workspaces/Financial-Distress-Predictor/models/optimal_features.pkl")

# URL of the FastAPI batch endpoint
url = "http://127.0.0.1:8000/predict_batch"

# Generate a batch of 5 sample companies with random feature values
batch_data = {
    "companies": []
}

for _ in range(5):
    company_features = {f: random.random() for f in optimal_features}
    batch_data["companies"].append({"features": company_features})

print("Sample batch input data:")
for i, c in enumerate(batch_data["companies"], 1):
    print(f"Company {i}:", c["features"])

# Send POST request to FastAPI
response = requests.post(url, json=batch_data)

# Print response
if response.status_code == 200:
    print("\nBatch prediction results:")
    for i, res in enumerate(response.json()["results"], 1):
        print(f"Company {i}:", res)
else:
    print("Error:", response.status_code, response.text)

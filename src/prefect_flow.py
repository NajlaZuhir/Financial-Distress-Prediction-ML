from prefect import flow, task
import subprocess
import os


PYTHON_PATH = r"C:\Users\Najla\Desktop\Git25_26\Financial-Distress-Predictor\env\Scripts\python.exe"

@task
def train_model():
    print("🚀 Training model...")
    subprocess.run([PYTHON_PATH, "src/train.py"], check=True)

@task
def log_to_mlflow():
    print("📊 Logging experiment to MLflow...")
    subprocess.run([PYTHON_PATH, "src/mlflow_integration.py"], check=True)



@flow(name="financial-distress-ml-pipeline")
def ml_pipeline():
    train_model()
    log_to_mlflow()

if __name__ == "__main__":
    ml_pipeline()

#Imports

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score



# Data preparation

def read_dataframe(filename):
    le = LabelEncoder()
    df = pd.read_csv(filename)

    df['Financial Distress'] = np.where(df['Financial Distress'] > -0.5, 'Distressed', 'Not Distressed')
    df['Financial Distress'] = le.fit_transform(df['Financial Distress'])

    return df


def split_data(X, y, test_size=0.4, val_size=0.5, random_state=42):

    # Step 1: Split into training and a temporary set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 2: Split the temporary set into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test



def scaling_data(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def feature_selection(model, X_train, X_train_scaled, X_val_scaled, y_train, y_val):
  feature_importance = pd.Series(
    np.abs(model.coef_[0]),
    index=X_train.columns
    ).sort_values(ascending=False)

  f1_scores = []
  accuracy_scores = []
  feature_counts = range(1, len(feature_importance) + 1)

  for k in feature_counts:
      top_features = feature_importance.index[:k]

      cols_idx = [X.columns.get_loc(f) for f in top_features]

      X_train_k = X_train_scaled[:, cols_idx]
      X_val_k = X_val_scaled[:, cols_idx]

      model = LogisticRegression(
          max_iter=1000,
          class_weight="balanced",
          solver="liblinear"
      )

      model.fit(X_train_k, y_train)
      y_pred = model.predict(X_val_k)

      f1_scores.append(f1_score(y_val, y_pred, average="macro"))
      accuracy_scores.append(accuracy_score(y_val, y_pred))

  results_df = pd.DataFrame({
    "num_features": feature_counts,
    "f1_macro": f1_scores,
    "accuracy": accuracy_scores
  })

  results_df["delta_f1"] = results_df["f1_macro"].diff()

  threshold = 0.005
  window = 5
  plateau_point = None

  for i in range(len(results_df) - window):
      if results_df["delta_f1"].iloc[i+1:i+window+1].abs().max() < threshold:
          plateau_point = results_df["num_features"].iloc[i]
          break

  optimal_features = feature_importance.index[:plateau_point]

  return optimal_features


# training

def train_model(X_train_scaled, y_train):

  model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
  )

  model.fit(X_train_scaled, y_train)

  return model

def predict_financial_distress(new_data_df, trained_model, optimal_features):
    X_new = new_data_df.loc[:, optimal_features]
    y_pred = trained_model.predict(X_new)
    return y_pred


# Evaluation 

def evaluate_model(model, X, y, dataset_name="Dataset"):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred)
    return acc, f1, cm, cr


if __name__ == "__main__":

  # =============================
  # Step 1: Load dataset
  # =============================
  filename = "/workspaces/Financial-Distress-Predictor/data/Financial Distress.csv"
  df = read_dataframe(filename)

  X = df.iloc[:, 3:]  
  y = df['Financial Distress']

  # =============================
  # Step 2: Split data
  # =============================
  X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.4, val_size=0.5)

  # =============================
  # Step 3: Scale data
  # =============================
  X_train_scaled, X_val_scaled, X_test_scaled, scaler = scaling_data(X_train, X_val, X_test)

  # =============================
  # Step 4: Train initial model
  # =============================
  model = train_model(X_train_scaled, y_train)

  # =============================
  # Step 5: Feature Selection (plateau-based)
  # =============================
  optimal_features = feature_selection(
      model,
      X_train,
      X_train_scaled,
      X_val_scaled,
      y_train,
      y_val
  )

  # =============================
  # Step 6: Retrain model on optimal features
  # =============================

  cols_idx = [X_train.columns.get_loc(f) for f in optimal_features]
  # Select columns BEFORE scaling
  X_train_opt = X_train[optimal_features]
  X_val_opt   = X_val[optimal_features]
  X_test_opt  = X_test[optimal_features]

  # Fit scaler ONLY on optimal features
  scaler = StandardScaler()
  X_train_opt_scaled = scaler.fit_transform(X_train_opt)
  X_val_opt_scaled   = scaler.transform(X_val_opt)
  X_test_opt_scaled  = scaler.transform(X_test_opt)

  # Train final model
  final_model = train_model(X_train_opt_scaled, y_train)

  # =============================
  # Step 7: Saving the Model
  # =============================
  joblib.dump(final_model, "/workspaces/Financial-Distress-Predictor/models/financial_distress_model.pkl")
  joblib.dump(scaler, "/workspaces/Financial-Distress-Predictor/models/scaler.pkl")
  joblib.dump(optimal_features, "/workspaces/Financial-Distress-Predictor/models/optimal_features.pkl")


  # optimal_features
  with open("/workspaces/Financial-Distress-Predictor/models/output.txt", "w") as f:
    for feature in optimal_features:
      f.write(feature + "\n") 
      


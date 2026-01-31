import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    recall_score,
    confusion_matrix
)
from xgboost import XGBClassifier


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/Financial Distress.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Utility functions
# -----------------------------
def is_distressed(x):
    """1 = Distressed, 0 = Healthy"""
    return 1 if x < -0.5 else 0


def read_data(path):
    return pd.read_csv(path)


def feature_engineering(df):
    # Drop highly correlated features
    corr_threshold = 0.85
    X = df.drop("Financial Distress", axis=1)

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > corr_threshold)
    ]

    df = df.drop(columns=to_drop)

    # Drop weakly correlated features
    cor_target = df.corr()["Financial Distress"].abs()
    low_corr_features = cor_target[cor_target < 0.05].index.tolist()
    df = df.drop(columns=low_corr_features)

    return df


def split_data(df):
    df["Financial Distress"] = df["Financial Distress"].apply(is_distressed)

    X = df.drop("Financial Distress", axis=1)
    y = df["Financial Distress"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def tune_xgboost(X_train, y_train):
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=RANDOM_STATE
    )

    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7]
    }

    f1_scorer = make_scorer(f1_score, pos_label=1)

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=50,
        scoring=f1_scorer,
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def find_optimal_threshold(model, X_val, y_val, min_recall=0.6):
    """
    Find threshold that maximizes F1 while maintaining minimum recall.
    This matches the notebook's threshold selection strategy.
    """
    probs = model.predict_proba(X_val)[:, 1]
    
    # Test range of thresholds (matching notebook)
    thresholds = np.arange(0.05, 0.51, 0.05)
    results = []
    
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        rec = recall_score(y_val, y_pred, pos_label=1)
        f1 = f1_score(y_val, y_pred, pos_label=1)
        results.append({
            'threshold': t,
            'recall': rec,
            'f1': f1
        })
    
    df_results = pd.DataFrame(results)
    
    print("\nThreshold Tuning Results:")
    print(df_results.to_string(index=False))
    
    # Filter for minimum recall, then maximize F1
    valid = df_results[df_results['recall'] >= min_recall]
    
    if len(valid) == 0:
        print(f"\n⚠️  Warning: No threshold achieves recall >= {min_recall}")
        print("Selecting threshold with maximum F1 instead.")
        optimal = df_results.loc[df_results['f1'].idxmax()]
    else:
        optimal = valid.loc[valid['f1'].idxmax()]
    
    optimal_threshold = optimal['threshold']
    
    print(f"\n✓ Optimal Threshold: {optimal_threshold}")
    print(f"  Recall at this threshold: {optimal['recall']:.4f}")
    print(f"  F1-score at this threshold: {optimal['f1']:.4f}")
    
    return optimal_threshold


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    # Load & preprocess
    print("Loading and preprocessing data...")
    df_raw = read_data(DATA_PATH)
    df_fe = feature_engineering(df_raw)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_fe)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"\nFinal feature set: {len(X_train.columns)} features")

    # Train model
    print("\n" + "="*60)
    print("Training XGBoost with RandomizedSearchCV...")
    print("="*60)
    model, best_params = tune_xgboost(X_train, y_train)
    
    print("\n✓ Best Parameters Found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Threshold tuning
    print("\n" + "="*60)
    print("Tuning Classification Threshold...")
    print("="*60)
    optimal_threshold = find_optimal_threshold(model, X_val, y_val, min_recall=0.6)

    # Validation evaluation with optimal threshold
    val_probs = model.predict_proba(X_val)[:, 1]
    y_val_pred = (val_probs >= optimal_threshold).astype(int)

    print("\n" + "="*60)
    print("VALIDATION RESULTS (with optimal threshold)")
    print("="*60)
    print(classification_report(y_val, y_val_pred, digits=4))

    print("Confusion Matrix (Validation):")
    cm = confusion_matrix(y_val, y_val_pred)
    print(cm)
    print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

    # Calculate key metrics
    minority_recall = recall_score(y_val, y_val_pred, pos_label=1)
    minority_f1 = f1_score(y_val, y_val_pred, pos_label=1)
    
    print(f"\n📊 Key Metrics:")
    print(f"  Minority Class Recall: {minority_recall:.4f}")
    print(f"  Minority Class F1-Score: {minority_f1:.4f}")

    # Save test set as CSV
    print("\n" + "="*60)
    print("Saving Test Set...")
    print("="*60)
    
    # Combine X_test and y_test into a single DataFrame
    test_df = X_test.copy()
    test_df['Financial Distress'] = y_test.values
    
    # Save to CSV
    test_csv_path = os.path.join(MODEL_DIR, "test.csv")
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"✓ Test set saved to: {test_csv_path}")
    print(f"  Total samples: {len(test_df)}")
    print(f"  Features: {len(X_test.columns)}")
    print(f"  Distressed: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
    print(f"  Healthy: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")

    # Save model artifacts
    final_features = X_train.columns.tolist()

    joblib.dump(model, f"{MODEL_DIR}/xgb_final_model.pkl")
    joblib.dump(final_features, f"{MODEL_DIR}/final_features.pkl")
    joblib.dump(optimal_threshold, f"{MODEL_DIR}/optimal_threshold.pkl")

    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Saved artifacts to `{MODEL_DIR}/`:")
    print(f"  - xgb_final_model.pkl")
    print(f"  - final_features.pkl")
    print(f"  - optimal_threshold.pkl (threshold={optimal_threshold})")
    print(f"  - test.csv (test set with {len(test_df)} samples)")
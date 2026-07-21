import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

# Load Data
df = pd.read_csv("synthetic_patients.csv")

features = ["age", "sex_encoded", "systolic_bp", "diastolic_bp", "bmi", "heart_rate", "random_glucose", "spo2"]
X = df[features]
y = df["risk_level"]

labels = ["low_risk", "moderate_risk", "high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter Optimization Grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}

base_model = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_grid,
    n_iter=15,
    cv=3,
    scoring='f1_macro',
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

# Probability Calibration
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_train, y_train)

# Evaluation
preds = calibrated_model.predict(X_test)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, preds, target_names=labels))

print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test, preds), index=labels, columns=labels))

# Save Best Model & Metadata
best_model.save_model("model.json")

with open("model_meta.json", "w") as f:
    json.dump({
        "model_version": "toy-v0.4-calibrated",
        "training_data_source": "synthetic-v4-ihci-risk",
        "features": features,
        "labels": labels,
        "best_params": search.best_params_
    }, f, indent=2)

print("\nSaved calibrated model.json and model_meta.json successfully.")
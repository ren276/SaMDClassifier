import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Load canonical dataset
df = pd.read_csv("dataset/canonical_dataset.csv")

# The canonical dataset does not contain glucose. Pulse is the closest match to
# the current API's heart_rate field, so we train on the shared vital-sign set.
df = df.assign(
    age=df["age_at_encounter"],
    sex_encoded=df["sex"].map({"M": 0, "F": 1}),
    systolic_bp=df["bp_systolic"],
    diastolic_bp=df["bp_diastolic"],
    heart_rate=df["pulse"],
)

features = ["age", "sex_encoded", "systolic_bp", "diastolic_bp", "bmi", "heart_rate", "spo2"]
X = df[features]

# Collapse the four canonical tiers into the three risk bands used by the API.
y = df["tier"].map({1: 0, 2: 1, 3: 2, 4: 2})

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
predicted_probs = calibrated_model.predict_proba(X_test)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, preds, target_names=labels))

print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test, preds), index=labels, columns=labels))
print("\n=== SUMMARY METRICS ===")
print(f"accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"f1_macro: {f1_score(y_test, preds, average='macro'):.4f}")

# Save Best Model & Metadata
best_model.save_model("model.json")

with open("model_meta.json", "w") as f:
    json.dump({
        "model_version": "toy-v0.5-canonical-tier-bridge",
        "training_data_source": "dataset/canonical_dataset.csv",
        "features": features,
        "labels": labels,
        "best_params": search.best_params_,
        "target_definition": {
            "source_column": "tier",
            "mapping": {"1": "low_risk", "2": "moderate_risk", "3": "high_risk", "4": "high_risk"}
        },
        "validation_metrics": {
            "accuracy": accuracy_score(y_test, preds),
            "f1_macro": f1_score(y_test, preds, average='macro')
        },
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "calibration_used_for_evaluation": True
    }, f, indent=2)

print("\nSaved calibrated model.json and model_meta.json successfully.")
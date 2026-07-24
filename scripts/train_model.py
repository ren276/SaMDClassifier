import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_VERSION = "toy-v0.6-observed-glucose-4tier"
TRAINING_DATA_SOURCE = os.path.join(PROJECT_ROOT, "dataset", "canonical_dataset.csv")

# Load canonical dataset
df = pd.read_csv(TRAINING_DATA_SOURCE)

# Train on the noised "_observed" columns (as-measured values), not the
# ground-truth columns -- that's the actual point of the pipeline's
# noise-injection step. age/sex have no _observed variant (not noised
# upstream) so those still come from the ground-truth columns. glucose is
# intentionally sparse (~35% notna, opportunistic screening, not imputed) --
# left as NaN and handled by XGBoost's native missing-value split routing,
# not dropped or mean-imputed, since dropping would bias training toward
# whichever subpopulation happened to get a glucose test.
df = df.assign(
    age=df["age_at_encounter"],
    sex_encoded=df["sex"].map({"M": 0, "F": 1}),
    systolic_bp=df["bp_systolic_observed"],
    diastolic_bp=df["bp_diastolic_observed"],
    bmi=df["bmi_observed"],
    heart_rate=df["pulse_observed"],
    spo2=df["spo2_observed"],
    glucose=df["glucose_observed"],
)

features = ["age", "sex_encoded", "systolic_bp", "diastolic_bp", "bmi", "heart_rate", "spo2", "glucose"]
X = df[features]

# Collapse the four canonical tiers (tier = count of abnormal params, tier 4
# = "3+ abnormal params") into the three risk bands used by the API. Tiers 3
# (6.5%) and 4 (1.5%) are merged into high_risk: tier 4 alone is too thin
# (~330 rows, ~66 in a 20% test split) for a stable per-class F1, and tiers
# 3/4 are clinically adjacent (2 vs 3+ simultaneous abnormal vitals).
tier_mapping = {1: "low_risk", 2: "moderate_risk", 3: "high_risk", 4: "high_risk"}
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

# K-fold CV report on the winning hyperparameters, independent of
# RandomizedSearchCV's internal cv=3 (which only scores during the search).
kfold_scores = cross_val_score(
    best_model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro'
)
print("\n=== 5-FOLD CV (f1_macro, post-search re-fit) ===")
print(kfold_scores)
print(f"mean: {kfold_scores.mean():.4f}  std: {kfold_scores.std():.4f}")

# Evaluation on the raw, uncalibrated model -- this is the model actually
# serialized below and served by app.py (xgb.XGBClassifier.load_model +
# shap.TreeExplainer both require a native XGBoost booster; a
# CalibratedClassifierCV wrapper can't be saved in that format or explained
# by TreeExplainer, so calibrating here would evaluate a different model
# than the one that ships). Real probability calibration needs an app.py
# load/explain rework -- deferred to that pass, not done silently here.
preds = best_model.predict(X_test)
predicted_probs = best_model.predict_proba(X_test)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, preds, target_names=labels))

print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test, preds), index=labels, columns=labels))

per_class_f1 = f1_score(y_test, preds, average=None)
print("\n=== SUMMARY METRICS ===")
print(f"accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"f1_macro: {f1_score(y_test, preds, average='macro'):.4f}")
for label, score in zip(labels, per_class_f1):
    print(f"f1[{label}]: {score:.4f}")

# Save Best Model & Metadata
best_model.save_model(os.path.join(PROJECT_ROOT, "models", "model.json"))

run_timestamp = datetime.now(timezone.utc).isoformat()

meta = {
    "model_version": MODEL_VERSION,
    "training_data_source": TRAINING_DATA_SOURCE,
    "features": features,
    "labels": labels,
    "best_params": search.best_params_,
    "target_definition": {
        "source_column": "tier",
        "mapping": {str(k): v for k, v in tier_mapping.items()}
    },
    "glucose_missing_strategy": "native_xgboost_nan_routing_no_imputation",
    "validation_metrics": {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average='macro'),
        "f1_per_class": dict(zip(labels, per_class_f1.tolist())),
        "kfold_f1_macro_mean": float(kfold_scores.mean()),
        "kfold_f1_macro_std": float(kfold_scores.std()),
        "kfold_f1_macro_scores": kfold_scores.tolist()
    },
    "rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "calibration_used_for_evaluation": False
}

with open(os.path.join(PROJECT_ROOT, "models", "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

with open(os.path.join(PROJECT_ROOT, "logs", "training_log.jsonl"), "a") as f:
    f.write(json.dumps({"timestamp": run_timestamp, **meta}) + "\n")

print("\nSaved model.json and model_meta.json, appended training_log.jsonl.")

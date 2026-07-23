import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

MODEL_VERSION = "symptom-clf-v0.1-tfidf-xgboost"
TRAINING_DATA_SOURCE = "dataset/canonical_dataset.csv"

# Load canonical dataset
df = pd.read_csv(TRAINING_DATA_SOURCE)

print("=== icd_candidate VALUE COUNTS (all rows) ===")
print(df["icd_candidate"].value_counts(dropna=False))

# icd_candidate is null on ~32% of rows by dataset design (combination doesn't
# narrow below ICD block level -- not an error). Verified separately that
# "no specific symptom" never co-occurs with a non-null icd_candidate, so
# this filter is a clean split, not discarding ambiguous/conflicting signal.
df = df[df["icd_candidate"].notna()].copy()

print("\n=== icd_candidate VALUE COUNTS (labeled subset used for training) ===")
class_counts = df["icd_candidate"].value_counts()
print(class_counts)

# Word-level (1,2-gram) + char-level (char_wb, 3-5 gram) TF-IDF, combined.
# Char n-grams add some robustness to misspellings/OOV variants, but the
# training data itself is sampled from a fixed per-condition symptom pool
# with zero spelling variance -- this does NOT validate the model against
# real-world noisy free text. See README "Known limitation".
word_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

X_word = word_vectorizer.fit_transform(df["symptom_string"])
X_char = char_vectorizer.fit_transform(df["symptom_string"])
X = hstack([X_word, X_char]).tocsr()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["icd_candidate"])
labels = label_encoder.classes_.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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
    n_jobs=1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

kfold_scores = cross_val_score(
    best_model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro'
)
print("\n=== 5-FOLD CV (f1_macro, post-search re-fit) ===")
print(kfold_scores)
print(f"mean: {kfold_scores.mean():.4f}  std: {kfold_scores.std():.4f}")

preds = best_model.predict(X_test)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, preds, target_names=labels, zero_division=0))

print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(confusion_matrix(y_test, preds), index=labels, columns=labels))

per_class_f1 = f1_score(y_test, preds, average=None, zero_division=0)
per_class_recall = classification_report(y_test, preds, target_names=labels, output_dict=True, zero_division=0)

print("\n=== SUMMARY METRICS ===")
print(f"accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"f1_macro: {f1_score(y_test, preds, average='macro'):.4f}")
for label, score in zip(labels, per_class_f1):
    recall = per_class_recall[label]["recall"]
    support = int(per_class_recall[label]["support"])
    flag = "  <<< NEAR-ZERO RECALL" if recall < 0.15 else ""
    print(f"f1[{label}]: {score:.4f}  recall: {recall:.4f}  support: {support}{flag}")

# Save Model & Vectorizers
best_model.save_model("symptom_model.json")
joblib.dump(word_vectorizer, "symptom_vectorizer_word.joblib")
joblib.dump(char_vectorizer, "symptom_vectorizer_char.joblib")
joblib.dump(label_encoder, "symptom_label_encoder.joblib")


def predict_ranked(symptom_text, model=best_model, word_vec=word_vectorizer, char_vec=char_vectorizer, encoder=label_encoder):
    """Vectorize symptom_text and return all classes ranked by probability, descending."""
    xw = word_vec.transform([symptom_text])
    xc = char_vec.transform([symptom_text])
    xin = hstack([xw, xc]).tocsr()
    probs = model.predict_proba(xin)[0]
    ranked = sorted(zip(encoder.classes_, probs), key=lambda p: p[1], reverse=True)
    return ranked


run_timestamp = datetime.now(timezone.utc).isoformat()

meta = {
    "model_version": MODEL_VERSION,
    "training_data_source": TRAINING_DATA_SOURCE,
    "row_filter": "icd_candidate.notna() -- rows without a disease-level label excluded (32.2% of full dataset, by dataset design)",
    "features": {
        "input_column": "symptom_string",
        "word_vectorizer": {"ngram_range": [1, 2], "vocab_size": len(word_vectorizer.vocabulary_)},
        "char_vectorizer": {"analyzer": "char_wb", "ngram_range": [3, 5], "vocab_size": len(char_vectorizer.vocabulary_)},
        "combined_feature_dim": X.shape[1]
    },
    "labels": labels,
    "best_params": search.best_params_,
    "validation_metrics": {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average='macro'),
        "f1_per_class": dict(zip(labels, per_class_f1.tolist())),
        "kfold_f1_macro_mean": float(kfold_scores.mean()),
        "kfold_f1_macro_std": float(kfold_scores.std()),
        "kfold_f1_macro_scores": kfold_scores.tolist()
    },
    "rows": int(len(df)),
    "train_rows": int(X_train.shape[0]),
    "test_rows": int(X_test.shape[0]),
    "calibration_used_for_evaluation": False
}

with open("symptom_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

with open("symptom_training_log.jsonl", "a") as f:
    f.write(json.dumps({"timestamp": run_timestamp, **meta}) + "\n")

print("\nSaved symptom_model.json, vectorizers, symptom_model_meta.json, appended symptom_training_log.jsonl.")

# Demo: ranked candidate output for a few test-set examples, showing the full
# shape (all classes + probabilities) that the future refinement/re-ranking
# stage will consume -- not just top-1.
print("\n=== EXAMPLE RANKED PREDICTIONS ===")
sample_texts = df["symptom_string"].drop_duplicates().sample(5, random_state=42).tolist()
for text in sample_texts:
    ranked = predict_ranked(text)
    print(f"\nsymptom_string: {text!r}")
    for label, prob in ranked:
        print(f"  {label}: {prob:.4f}")

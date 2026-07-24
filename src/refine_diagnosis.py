"""
Stage 2: combines Classifier A (vitals -> risk tier) and Classifier B
(symptom_string -> ranked icd_candidate) into one recalibrated ranked
differential for physician review. Never collapses to a single winner --
output is always a top-3-to-5 ranked list.

Re-ranking method (see readme.md "Classifier B" / Stage 2 section for the full
write-up): a weighted blend, not a secondary ML model and not discrete
if/else rules. Each icd_candidate's symptom-model probability is multiplied by
a "vitals_tier_alignment" score -- how well that candidate's typical vitals-
severity profile (empirically read from the training dataset) matches what
Classifier A actually predicted for this patient's vitals -- then renormalized
across all classes. This keeps every adjustment traceable to two numbers per
candidate (symptom probability, alignment score) via one formula, with no
opaque model layer and no candidate ever multiplied to exactly zero.

This module also defines (schema only) the DiagnosisFeedback data contract
for a future physician-feedback capture step -- no live capture, persistence,
or export/re-import pipeline is wired up in this pass.
"""

import json
from datetime import datetime, timezone
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel
from scipy.sparse import hstack

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Load Classifier A (vitals -> risk tier) ---
with open(os.path.join(PROJECT_ROOT, "models", "model_meta.json")) as f:
    tier_meta = json.load(f)

tier_model = xgb.XGBClassifier()
tier_model.load_model(os.path.join(PROJECT_ROOT, "models", "model.json"))
tier_features = tier_meta["features"]
tier_labels = tier_meta["labels"]  # order matches predict_proba column order
tier_mapping = tier_meta["target_definition"]["mapping"]  # {"1": "low_risk", ...}

# --- Load Classifier B (symptom_string -> ranked icd_candidate) ---
with open(os.path.join(PROJECT_ROOT, "models", "symptom_model_meta.json")) as f:
    symptom_meta = json.load(f)

symptom_model = xgb.XGBClassifier()
symptom_model.load_model(os.path.join(PROJECT_ROOT, "models", "symptom_model.json"))
word_vectorizer = joblib.load(os.path.join(PROJECT_ROOT, "models", "symptom_vectorizer_word.joblib"))
char_vectorizer = joblib.load(os.path.join(PROJECT_ROOT, "models", "symptom_vectorizer_char.joblib"))
label_encoder = joblib.load(os.path.join(PROJECT_ROOT, "models", "symptom_label_encoder.joblib"))
icd_labels = label_encoder.classes_.tolist()


def _build_tier_profile():
    """For each icd_candidate, its empirical distribution over Classifier A's
    3 risk labels (same tier collapse Classifier A trains on, read from
    model_meta.json so this can't drift out of sync with Classifier A)."""
    df = pd.read_csv(os.path.join(PROJECT_ROOT, symptom_meta["training_data_source"]))
    df = df[df["icd_candidate"].notna()].copy()
    df["risk_label"] = df["tier"].astype(str).map(tier_mapping)

    profile = {}
    for candidate, group in df.groupby("icd_candidate"):
        counts = group["risk_label"].value_counts(normalize=True)
        profile[candidate] = np.array([counts.get(label, 0.0) for label in tier_labels])
    return profile


TIER_PROFILE = _build_tier_profile()


def get_symptom_ranked(symptom_string):
    """Full 18-class (icd_candidate, probability) list, sorted descending."""
    xw = word_vectorizer.transform([symptom_string])
    xc = char_vectorizer.transform([symptom_string])
    xin = hstack([xw, xc]).tocsr()
    probs = symptom_model.predict_proba(xin)[0]
    return sorted(zip(icd_labels, probs), key=lambda p: p[1], reverse=True)


def get_vitals_tier(vitals: dict):
    """Returns (predicted_label, {label: probability}) from Classifier A."""
    row = pd.DataFrame([{feature: vitals[feature] for feature in tier_features}])
    probs = tier_model.predict_proba(row)[0]
    predicted_label = tier_labels[int(np.argmax(probs))]
    return predicted_label, dict(zip(tier_labels, probs))


def refine(symptom_string, vitals: dict, top_k: int = 5):
    """Re-ranks Classifier B's full candidate list using Classifier A's vitals
    signal. Always returns at least 3 candidates, never a single winner."""
    top_k = max(top_k, 3)

    symptom_ranked = get_symptom_ranked(symptom_string)
    symptom_prob_by_class = dict(symptom_ranked)
    symptom_rank_by_class = {c: i + 1 for i, (c, _) in enumerate(symptom_ranked)}

    predicted_tier, tier_probs_by_label = get_vitals_tier(vitals)
    tier_prob_vector = np.array([tier_probs_by_label[label] for label in tier_labels])

    adjusted = {}
    alignment_by_class = {}
    for candidate in icd_labels:
        profile = TIER_PROFILE.get(candidate, np.full(len(tier_labels), 1 / len(tier_labels)))
        alignment = float(np.dot(profile, tier_prob_vector))
        multiplier = 0.5 + alignment
        adjusted[candidate] = symptom_prob_by_class[candidate] * multiplier
        alignment_by_class[candidate] = alignment

    total = sum(adjusted.values())
    adjusted_confidence = {c: v / total for c, v in adjusted.items()}

    ranked = sorted(adjusted_confidence.items(), key=lambda p: p[1], reverse=True)[:top_k]

    results = []
    for new_rank, (candidate, adj_conf) in enumerate(ranked, start=1):
        original_conf = symptom_prob_by_class[candidate]
        original_rank = symptom_rank_by_class[candidate]
        alignment = alignment_by_class[candidate]
        profile = TIER_PROFILE.get(candidate)
        profile_str = ", ".join(f"{label}={p:.0%}" for label, p in zip(tier_labels, profile))
        direction = "boosted" if adj_conf > original_conf else "penalized" if adj_conf < original_conf else "unchanged"
        movement = f", moved #{original_rank} -> #{new_rank}" if original_rank != new_rank else ""

        why = (
            f"Symptom model ranked {candidate} #{original_rank} ({original_conf:.1%}). "
            f"Classifier A predicts {predicted_tier}; {candidate} typically presents at "
            f"[{profile_str}] -- alignment {alignment:.2f}, confidence {direction} to "
            f"{adj_conf:.1%}{movement}."
        )

        results.append({
            "icd_candidate": candidate,
            "adjusted_confidence": adj_conf,
            "original_symptom_confidence": original_conf,
            "vitals_tier_alignment": alignment,
            "why": why
        })

    return results


class VitalsInput(BaseModel):
    """Mirrors app.py's PatientVitalsRequest field names for consistency."""
    age: int
    sex: str
    systolic_bp: float
    diastolic_bp: float
    bmi: float
    heart_rate: float
    random_glucose: Optional[float] = None
    spo2: float


class DiagnosisFeedback(BaseModel):
    """
    Data contract for physician feedback on a Stage 2 refined differential.
    SCHEMA ONLY this pass -- no capture endpoint, no persistence, no
    export/re-import into drishti_pipeline is wired up yet. That's the
    explicit next step once this shape has been reviewed.

    Only physician_decision in {AGREE, MODIFY} will ever be eligible for a
    future dataset re-import (confirmed cases only). REJECT and any
    unconfirmed case must never be reimported -- there is no ground-truth
    diagnosis to trust in either case.
    """
    case_token: str
    symptom_string: str
    vitals: VitalsInput
    refined_candidates: list
    physician_decision: Literal["AGREE", "MODIFY", "REJECT"]
    physician_final_diagnosis: Optional[str] = None
    timestamp: str


if __name__ == "__main__":
    df = pd.read_csv(symptom_meta["training_data_source"])
    df = df[df["icd_candidate"].notna()].copy()

    def row_to_vitals(row):
        return {
            "age": row["age_at_encounter"],
            "sex_encoded": 0 if row["sex"] == "M" else 1,
            "systolic_bp": row["bp_systolic_observed"],
            "diastolic_bp": row["bp_diastolic_observed"],
            "bmi": row["bmi_observed"],
            "heart_rate": row["pulse_observed"],
            "spo2": row["spo2_observed"],
            "glucose": row["glucose_observed"]
        }

    clear_cut_classes = ["A01.0", "M17", "E66"]
    ambiguous_classes = ["A91", "I10", "J22"]

    demo_rows = []
    for cls in clear_cut_classes:
        demo_rows.append(("clear-cut", df[df["icd_candidate"] == cls].sample(1, random_state=1).iloc[0]))
    for cls in ambiguous_classes:
        demo_rows.append(("ambiguous", df[df["icd_candidate"] == cls].sample(1, random_state=1).iloc[0]))

    for case_type, row in demo_rows:
        symptom_string = row["symptom_string"]
        vitals = row_to_vitals(row)
        true_label = row["icd_candidate"]
        true_tier = row["tier"]

        raw_ranked = get_symptom_ranked(symptom_string)[:5]
        refined_ranked = refine(symptom_string, vitals, top_k=5)

        print(f"\n{'=' * 80}")
        print(f"[{case_type}] true icd_candidate={true_label} (tier={true_tier})")
        print(f"symptom_string: {symptom_string!r}")
        print(f"vitals: {vitals}")

        print("\nRAW Classifier B top-5:")
        for label, prob in raw_ranked:
            print(f"  {label}: {prob:.1%}")

        print("\nREFINED (Stage 2) top-5:")
        for r in refined_ranked:
            print(f"  {r['icd_candidate']}: {r['adjusted_confidence']:.1%}  (alignment={r['vitals_tier_alignment']:.2f})")
            print(f"    why: {r['why']}")

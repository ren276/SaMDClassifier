"""
Glues refine_diagnosis (Stage 2 ranking) and rag/rag_pipeline (NLEM 2022
cited recommendation) into one unified report, enriched with Indian brand
names and vitals-based triage. Purely deterministic routing/enrichment --
no generative model calls anywhere in this module.
"""

import os
import sys

_RAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag")
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from rag_pipeline import get_treatment_recommendation, ICD_TO_DISEASE_NAME  # noqa: E402

from indian_brands import enrich_with_indian_brands
from thresholds import combine_vitals

NLEM_TREATMENT_KEYS = (
    "recommendedDrug",
    "levelOfHealthcare",
    "availableAtPHC",
    "pediatricDose",
    "citation",
    "confidence",
    "referralReason",
    "matchedDisease",
)

TRIAGE_VITALS_KEYS = ("systolic_bp", "diastolic_bp", "pulse", "respiratory_rate", "spo2", "temperature", "bmi")


def execute_full_clinical_pipeline(refinement_output: list, patient_age: int, vitals: dict) -> dict:
    """
    refinement_output: the list returned by refine_diagnosis.refine().
    patient_age: patient age in years.
    vitals: dict shaped for thresholds.combine_vitals's kwargs -- distinct
        from the vitals dict refine_diagnosis.refine() itself expects (see
        app.py, which builds both shapes from the same request payload).
    """
    top_candidate = refinement_output[0] if refinement_output else {}
    primary_icd = top_candidate.get("icd_candidate")
    primary_ailment = ICD_TO_DISEASE_NAME.get(primary_icd, primary_icd)

    diagnostic_summary = {
        "primary_icd_candidate": primary_icd,
        "primary_ailment_name": primary_ailment,
        "differential": refinement_output,
    }

    rag_result = get_treatment_recommendation(refinement_output, patient_age)

    nlem_treatment = {key: rag_result.get(key) for key in NLEM_TREATMENT_KEYS}
    nlem_treatment["dosageForms"] = rag_result.get("dosageForms") or []

    recommended_drug = rag_result.get("recommendedDrug")
    brand_mapping = enrich_with_indian_brands(recommended_drug) if recommended_drug else None

    vitals_triage = None
    if vitals.get("respiratory_rate") is not None and vitals.get("temperature") is not None:
        vitals_triage = combine_vitals(
            **{key: vitals[key] for key in TRIAGE_VITALS_KEYS},
            glucose=vitals.get("glucose"),
            glucose_type=vitals.get("glucose_type", "fasting"),
            age=vitals.get("age"),
            sex=vitals.get("sex"),
        )

    safety_and_triage = {
        "vitals_triage": vitals_triage,
        "requiresHumanReview": rag_result.get("requiresHumanReview"),
        "pediatric_referral_flag": rag_result.get("pediatric_referral_flag"),
        "failure_reason": rag_result.get("failure_reason"),
    }

    return {
        "diagnostic_summary": diagnostic_summary,
        "nlem_treatment": nlem_treatment,
        "brand_mapping": brand_mapping,
        "safety_and_triage": safety_and_triage,
    }

"""
Manual integration checks for the glue pipeline (matches test_api.py's
plain-assert style -- not a pytest suite). Run directly:
    python test_integration.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import refine_diagnosis
import pipeline_glue
from indian_brands import enrich_with_indian_brands
from api_schemas import KernelReportOutput


def case_a_brand_enrichment():
    print("=== Case A: brand enrichment ===")
    r = enrich_with_indian_brands("  Amlodipine ")
    assert r["brand_mapping_available"] is True
    assert r["jan_aushadhi_brand"] == "Jan Aushadhi Amlodipine"
    assert r["commercial_brands"] == ["Amlopress", "Stamlo", "Amlopin"]

    r2 = enrich_with_indian_brands("Some Unmapped Drug")
    assert r2 == {
        "generic_name": "Some Unmapped Drug",
        "jan_aushadhi_brand": None,
        "commercial_brands": [],
        "brand_mapping_available": False,
    }
    print("PASS")


def case_b_valid_end_to_end_i10():
    print("=== Case B: valid end-to-end (I10 hypertension) ===")
    symptom_string = "no symptoms | epistaxis | blurred vision"  # real in-distribution I10 sample

    refine_vitals = {
        "age": 43, "sex_encoded": 0, "systolic_bp": 150, "diastolic_bp": 92,
        "bmi": 27.5, "heart_rate": 78, "spo2": 97, "glucose": 110,
    }
    triage_vitals = {
        "systolic_bp": 150, "diastolic_bp": 92, "pulse": 78, "respiratory_rate": 16,
        "spo2": 97, "temperature": 37.0, "bmi": 27.5, "glucose": 110,
        "glucose_type": "fasting", "age": 43, "sex": "M",
    }

    refinement_output = refine_diagnosis.refine(symptom_string, refine_vitals)
    assert len(refinement_output) >= 3

    result = pipeline_glue.execute_full_clinical_pipeline(refinement_output, 43, triage_vitals)
    report = KernelReportOutput(**result)  # must not raise

    assert len(report.diagnostic_summary.differential) >= 3

    treatment = report.nlem_treatment
    if treatment.recommendedDrug is not None:
        assert len(treatment.dosageForms) > 0
        assert treatment.citation is not None
    if report.brand_mapping is not None:
        assert report.brand_mapping.generic_name == treatment.recommendedDrug

    assert report.safety_and_triage.vitals_triage is not None
    assert report.safety_and_triage.vitals_triage.overall_urgency in {"urgent-review", "monitor", "routine"}

    print(f"primary_icd_candidate={report.diagnostic_summary.primary_icd_candidate}")
    print(f"recommendedDrug={treatment.recommendedDrug}")
    print(f"brand_mapping={report.brand_mapping}")
    print("PASS")


def case_c_adversarial_unknown_icd():
    print("=== Case C: adversarial out-of-vocabulary ICD ===")
    fake_refinement_output = [
        {"icd_candidate": "UNKNOWN_ICD", "adjusted_confidence": 0.5,
         "original_symptom_confidence": 0.5, "vitals_tier_alignment": 0.5,
         "why": "synthetic adversarial test case"},
        {"icd_candidate": "I10", "adjusted_confidence": 0.3,
         "original_symptom_confidence": 0.3, "vitals_tier_alignment": 0.3, "why": "..."},
        {"icd_candidate": "E11", "adjusted_confidence": 0.2,
         "original_symptom_confidence": 0.2, "vitals_tier_alignment": 0.2, "why": "..."},
    ]
    minimal_triage_vitals = {
        "systolic_bp": 120, "diastolic_bp": 80, "pulse": 70, "respiratory_rate": None,
        "spo2": 98, "temperature": None, "bmi": 22.0, "glucose": None,
        "glucose_type": "fasting", "age": 30, "sex": "F",
    }

    result = pipeline_glue.execute_full_clinical_pipeline(fake_refinement_output, 30, minimal_triage_vitals)
    report = KernelReportOutput(**result)  # must not raise

    assert report.nlem_treatment.recommendedDrug is None
    assert report.brand_mapping is None
    assert report.safety_and_triage.requiresHumanReview is True
    assert "unrecognized" in report.safety_and_triage.failure_reason
    assert report.safety_and_triage.vitals_triage is None  # respiratory_rate/temperature withheld

    print(f"failure_reason={report.safety_and_triage.failure_reason}")
    print("PASS")


if __name__ == "__main__":
    case_a_brand_enrichment()
    case_b_valid_end_to_end_i10()
    case_c_adversarial_unknown_icd()
    print("\nAll integration checks passed.")

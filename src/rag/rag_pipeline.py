"""
Orchestrates the NLEM 2022 RAG pipeline: takes the Stage 2 disease-ranking
output (see refine_diagnosis.refine) and a patient age, and returns a
validated, cited drug recommendation or a fallback referral. No LLM calls
anywhere in this path -- retrieval is local sentence-transformer embeddings,
extraction and validation are deterministic regex/lookup logic.
"""

from retriever import retrieve
from extractor import extract_recommendation
from validator import validate_output

# Closed, static lookup for the exact 18 icd_candidate labels Classifier B
# is trained on (symptom_model_meta.json["labels"]). This is a fixed table,
# not a model call, so it carries no black-box risk.
ICD_TO_DISEASE_NAME = {
    "A01.0": "Typhoid fever",
    "A09": "Infectious gastroenteritis and colitis",
    "A15": "Respiratory tuberculosis",
    "A90": "Dengue fever",
    "A91": "Dengue haemorrhagic fever",
    "A92.0": "Chikungunya fever",
    "B50": "Plasmodium falciparum malaria",
    "B54": "Unspecified malaria",
    "D50": "Iron deficiency anaemia",
    "E05.9": "Thyrotoxicosis (hyperthyroidism)",
    "E11": "Type 2 diabetes mellitus",
    "E66": "Obesity",
    "F41.0": "Panic disorder",
    "G43.9": "Migraine",
    "I10": "Essential hypertension",
    "J22": "Acute lower respiratory infection",
    "M17": "Osteoarthritis of the knee",
    "N39.0": "Urinary tract infection",
}


def get_treatment_recommendation(disease_ranking_output: list, age: int) -> dict:
    """
    disease_ranking_output: the list returned by refine_diagnosis.refine(),
        e.g. [{"icd_candidate": "I10", "adjusted_confidence": 0.62, ...}, ...]
    age: patient age in years.
    """
    if not disease_ranking_output:
        empty_extraction = {
            "recommendedDrug": None,
            "levelOfHealthcare": None,
            "availableAtPHC": None,
            "dosageForms": [],
            "pediatricDose": None,
            "citation": None,
            "confidence": "low",
            "referralReason": "No disease candidates were provided by Stage 2 ranking.",
        }
        return validate_output(empty_extraction, [], age)

    top_candidate = disease_ranking_output[0]
    icd_code = top_candidate.get("icd_candidate")
    
    if icd_code not in ICD_TO_DISEASE_NAME:
        empty_extraction = {
            "recommendedDrug": None,
            "levelOfHealthcare": None,
            "availableAtPHC": None,
            "dosageForms": [],
            "pediatricDose": None,
            "citation": None,
            "confidence": "low",
            "referralReason": f"Disease candidate '{icd_code}' is unrecognized and not supported by the NLEM 2022 corpus.",
        }
        return validate_output(empty_extraction, [], age)

    disease_name = ICD_TO_DISEASE_NAME.get(icd_code)

    retrieved_chunks = retrieve(disease_name, top_k=5)
    extracted = extract_recommendation(top_candidate, retrieved_chunks, age)
    validated = validate_output(extracted, retrieved_chunks, age)

    validated["matchedDisease"] = {"icd_candidate": icd_code, "disease_name": disease_name}
    return validated

"""
Deterministic post-extraction validation for the NLEM 2022 RAG pipeline.
No LLM: every check is a structural/string comparison against data already
parsed at ingestion time. Any failed check (or a null extraction) sets
requiresHumanReview=True with a recorded failure_reason. Age < 18 always
forces requiresHumanReview=True regardless of how the other checks fared --
this is a hard-fail invariant, not a soft signal.
"""

import re

from retriever import get_alpha_index_entries


def _normalize_name(name: str) -> str:
    name = re.sub(r"\*+$", "", name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name.lower()


def _drug_exists(drug: str, retrieved_chunks: list) -> bool:
    norm_drug = _normalize_name(drug)
    for entry in get_alpha_index_entries():
        if _normalize_name(entry.get("name", "")) == norm_drug:
            return True
    # Secondary check: fall back to the retrieved chunk text itself.
    for chunk in retrieved_chunks:
        if norm_drug in _normalize_name(chunk.get("text", "")):
            return True
    return False


def validate_output(extracted_json: dict, retrieved_chunks: list, age: int) -> dict:
    result = dict(extracted_json)
    failure_reasons = []
    requires_human_review = False

    drug = result.get("recommendedDrug")
    if drug is None:
        requires_human_review = True
        failure_reasons.append(
            result.get("referralReason")
            or "No drug recommendation could be extracted from NLEM 2022 for this candidate diagnosis."
        )
    else:
        if not _drug_exists(drug, retrieved_chunks):
            requires_human_review = True
            failure_reasons.append(
                f"'{drug}' could not be verified against the NLEM 2022 alphabetical medicine index."
            )

        dosage_forms = result.get("dosageForms")
        if not isinstance(dosage_forms, list) or not dosage_forms:
            requires_human_review = True
            failure_reasons.append("Extracted dosage form(s) are missing or invalid.")

        citation = result.get("citation")
        if not citation or not citation.get("page") or not citation.get("item_num"):
            requires_human_review = True
            failure_reasons.append("Citation to NLEM 2022 source is missing or incomplete.")

    pediatric_referral_flag = age < 18
    if pediatric_referral_flag:
        if result.get("pediatricDose") is not None:
            requires_human_review = True
            failure_reasons.append(
                "Pediatric dose was unexpectedly populated; deterministic extraction does "
                "not support pediatric dosing -- requires human review."
            )
        requires_human_review = True
        failure_reasons.append(
            "Patient is under 18 -- pediatric cases always require human review before "
            "any NLEM-derived recommendation is acted on."
        )

    result["pediatric_referral_flag"] = pediatric_referral_flag
    result["requiresHumanReview"] = requires_human_review
    result["failure_reason"] = "; ".join(failure_reasons) if failure_reasons else None
    return result

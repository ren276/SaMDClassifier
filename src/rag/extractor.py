"""
Deterministic (no-LLM) extraction of a structured drug recommendation from
NLEM 2022 chunks already retrieved by rag/retriever.py. All fields come
straight from the regex-parsed chunk metadata produced during ingestion --
nothing here is inferred or generated.
"""

import re

# Cosine distance from Chroma (0 = identical, 2 = opposite). Above this the
# retrieved chunk is not considered a reliable match for the disease query.
DISTANCE_THRESHOLD = 0.6


def _strip_footnote_markers(name: str) -> str:
    """Drop trailing NLEM footnote markers (*, **, #) from a medicine name
    for display; the underlying chunk text/citation still carries the raw
    form for traceability."""
    return re.sub(r"[\*#\s]+$", "", name)


def _null_recommendation(reason: str) -> dict:
    return {
        "recommendedDrug": None,
        "levelOfHealthcare": None,
        "availableAtPHC": None,
        "dosageForms": [],
        "pediatricDose": None,
        "citation": None,
        "confidence": "low",
        "referralReason": reason,
    }


def extract_recommendation(disease_ranking: dict, retrieved_chunks: list, age: int) -> dict:
    """Pick the best NLEM chunk and structure it into a recommendation dict.

    Returns a `recommendedDrug: None` fallback result if no chunk is
    available, the best match is not semantically close enough, or the
    chunk's parsed structure is incomplete.
    """
    if not retrieved_chunks:
        return _null_recommendation("No NLEM 2022 chunks retrieved for this disease.")

    best = retrieved_chunks[0]
    distance = best.get("distance")
    if distance is not None and distance > DISTANCE_THRESHOLD:
        return _null_recommendation(
            f"Best NLEM match too weak (distance={distance:.3f} > {DISTANCE_THRESHOLD})."
        )

    name = (best.get("name") or "").strip()
    level = (best.get("level") or "").strip()
    level_codes = [c.strip() for c in level.split(",") if c.strip()]
    dosage_lines = [d.strip() for d in (best.get("dosage_lines") or "").split(" | ") if d.strip()]

    if not name or not level_codes or not dosage_lines:
        return _null_recommendation("Retrieved NLEM chunk is missing required fields.")

    name = _strip_footnote_markers(name)

    available_at_phc = "P" in level_codes
    citation = {
        "source": "NLEM 2022",
        "page": best.get("page"),
        "section": f"Section {best.get('section_num')} - {best.get('section_title')}",
        "subsection": best.get("subsection_title") or None,
        "item_num": best.get("item_num"),
    }

    referral_reason = None
    if not available_at_phc:
        referral_reason = (
            f"{name} is listed at {'/'.join(level_codes)} level only per NLEM 2022 -- "
            "not available at Primary Health Centre (PHC). Refer to higher facility."
        )

    return {
        "recommendedDrug": name,
        "levelOfHealthcare": level_codes,
        "availableAtPHC": available_at_phc,
        "dosageForms": dosage_lines,
        "pediatricDose": None,
        "citation": citation,
        "confidence": "high",
        "referralReason": referral_reason,
    }

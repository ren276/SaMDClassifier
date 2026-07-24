"""
Static generic-to-brand mapping for Indian Jan Aushadhi + commercial brand
names. Deterministic lookup only -- no inference, no LLM. Unknown drugs get
a safe fallback dict so the Android client never crashes on a missing key.
"""

INDIAN_BRAND_MAP = {
    "paracetamol": {
        "jan_aushadhi_brand": "Jan Aushadhi Paracetamol",
        "commercial_brands": ["Dolo-650", "Calpol", "Crocin"],
    },
    "amlodipine": {
        "jan_aushadhi_brand": "Jan Aushadhi Amlodipine",
        "commercial_brands": ["Amlopress", "Stamlo", "Amlopin"],
    },
    "telmisartan": {
        "jan_aushadhi_brand": "Jan Aushadhi Telmisartan",
        "commercial_brands": ["Telma", "Telmikind", "Telsartan"],
    },
    "amoxicillin": {
        "jan_aushadhi_brand": "Jan Aushadhi Amoxicillin",
        "commercial_brands": ["Mox", "Novamox", "Amoxil"],
    },
    "metformin": {
        "jan_aushadhi_brand": "Jan Aushadhi Metformin",
        "commercial_brands": ["Glycomet", "Glyciphage"],
    },
    "azithromycin": {
        "jan_aushadhi_brand": "Jan Aushadhi Azithromycin",
        "commercial_brands": ["Azee", "Azithral"],
    },
}


def enrich_with_indian_brands(generic_drug_name: str) -> dict:
    """Exact (case/whitespace-insensitive) lookup only -- a combination or
    salt-form name that doesn't match one of the mapped generics safely
    falls back rather than guessing."""
    key = (generic_drug_name or "").strip().lower()
    entry = INDIAN_BRAND_MAP.get(key)
    if entry is None:
        return {
            "generic_name": generic_drug_name,
            "jan_aushadhi_brand": None,
            "commercial_brands": [],
            "brand_mapping_available": False,
        }
    return {
        "generic_name": generic_drug_name,
        "jan_aushadhi_brand": entry["jan_aushadhi_brand"],
        "commercial_brands": entry["commercial_brands"],
        "brand_mapping_available": True,
    }

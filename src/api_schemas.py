"""
Strict Pydantic API contract for the Android client. Mirrors exactly what
pipeline_glue.execute_full_clinical_pipeline() returns. Optional fields are
used aggressively wherever a fallback/referral case can legitimately null
them out -- validation must never fail just because the deterministic
pipeline correctly declined to recommend a drug.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict


class RankedCandidate(BaseModel):
    icd_candidate: str
    adjusted_confidence: float
    original_symptom_confidence: float
    vitals_tier_alignment: float
    why: str
    # Same parts as `why`, built without a digit or "%": safe to show a community worker, who
    # must not be shown a model score. `why` embeds the scores and is for physicians.
    why_plain: str


class DiagnosticSummary(BaseModel):
    primary_icd_candidate: Optional[str] = None
    primary_ailment_name: Optional[str] = None
    differential: list[RankedCandidate] = []


class Citation(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    item_num: Optional[str] = None


class MatchedDisease(BaseModel):
    icd_candidate: str
    disease_name: str


class NlemTreatment(BaseModel):
    recommendedDrug: Optional[str] = None
    levelOfHealthcare: Optional[list[str]] = None
    availableAtPHC: Optional[bool] = None
    dosageForms: list[str] = []
    pediatricDose: Optional[str] = None
    citation: Optional[Citation] = None
    confidence: Optional[str] = None
    referralReason: Optional[str] = None
    matchedDisease: Optional[MatchedDisease] = None


class BrandMapping(BaseModel):
    generic_name: str
    jan_aushadhi_brand: Optional[str] = None
    commercial_brands: list[str] = []
    brand_mapping_available: bool


class VitalsTriage(BaseModel):
    bp_grade: str
    pulse: str
    respiratory_rate: str
    spo2: str
    temperature: str
    bmi: str
    glucose: Optional[str] = None
    overall_urgency: str


class SafetyAndTriage(BaseModel):
    vitals_triage: Optional[VitalsTriage] = None
    requiresHumanReview: bool
    pediatric_referral_flag: bool
    failure_reason: Optional[str] = None


class CalibratedFlags(BaseModel):
    vitals: Optional[bool] = None
    symptom: Optional[bool] = None


class EvaluateModelMetadata(BaseModel):
    """Identity of everything that produced this result. Hashes are of the artifacts the app
    verified against ARTIFACT_MANIFEST.json at start-up. `calibrated` is read from each
    artifact's metadata: None means the artifact does not say."""

    vitals_model_version: str
    vitals_model_sha256: str
    symptom_model_version: str
    symptom_model_sha256: str
    calibrated: CalibratedFlags
    embedding_model: str


class KernelReportOutput(BaseModel):
    # `model_metadata` is a wire name shared with /v1/assess; pydantic reserves the `model_` prefix.
    model_config = ConfigDict(protected_namespaces=())

    diagnostic_summary: DiagnosticSummary
    nlem_treatment: NlemTreatment
    brand_mapping: Optional[BrandMapping] = None
    safety_and_triage: SafetyAndTriage
    model_metadata: Optional[EvaluateModelMetadata] = None

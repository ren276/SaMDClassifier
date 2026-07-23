"""
Strict Pydantic API contract for the Android client. Mirrors exactly what
pipeline_glue.execute_full_clinical_pipeline() returns. Optional fields are
used aggressively wherever a fallback/referral case can legitimately null
them out -- validation must never fail just because the deterministic
pipeline correctly declined to recommend a drug.
"""

from typing import Optional

from pydantic import BaseModel


class RankedCandidate(BaseModel):
    icd_candidate: str
    adjusted_confidence: float
    original_symptom_confidence: float
    vitals_tier_alignment: float
    why: str


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


class KernelReportOutput(BaseModel):
    diagnostic_summary: DiagnosticSummary
    nlem_treatment: NlemTreatment
    brand_mapping: Optional[BrandMapping] = None
    safety_and_triage: SafetyAndTriage

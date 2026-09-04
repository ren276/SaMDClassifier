import json
import logging

# `pipeline_glue` pulls in sentence-transformers -> torch (RAG treatment lookup). On Windows,
# loading torch's bundled OpenMP runtime AFTER xgboost/shap have already initialized theirs
# corrupts DLL init state (OSError WinError 1114 loading c10.dll). Import it first so torch's
# runtime initializes cleanly before xgboost/shap load their own.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline_glue

import xgboost as xgb
import shap
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

import refine_diagnosis
from api_schemas import KernelReportOutput

logger = logging.getLogger(__name__)

import os

# Load Model & Metadata
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(PROJECT_ROOT, "models", "model_meta.json"), "r") as f:
        meta = json.load(f)
    
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(PROJECT_ROOT, "models", "model.json"))
    explainer = shap.TreeExplainer(model)
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

app = FastAPI(title="SaMD PHC Classifier Engine", version="1.0")

class PatientVitalsRequest(BaseModel):
    case_token: str
    age: int
    sex: str
    systolic_bp: float
    diastolic_bp: float
    bmi: float
    heart_rate: float
    random_glucose: Optional[float] = None
    spo2: float

@app.post("/v1/assess")
async def assess_patient(payload: PatientVitalsRequest):
    # 1. Deterministic Safety Gate (Red Flags)
    if payload.spo2 < 90 or payload.systolic_bp >= 180 or payload.diastolic_bp >= 110:
        return {
            "case_token": payload.case_token,
            "safety_screen_passed": False,
            "triage_urgency": "EMERGENCY_REFERRAL",
            "differential_diagnosis": [
                {
                    "condition_tier": "critical_vitals_flag",
                    "probability": 1.0,
                    "evidence_for": [f"Critical Threshold Breach: SpO2 {payload.spo2}%, Systolic BP {payload.systolic_bp} mmHg, Diastolic BP {payload.diastolic_bp} mmHg"],
                    "evidence_against": []
                }
            ],
            "recommended_investigations": ["Immediate Oxygen Therapy", "Emergency Medical Transfer"],
            "model_metadata": {"version": meta["model_version"], "calibrated": True}
        }
        
    # 2. Vectorize Data for XGBoost
    sex_encoded = 0 if payload.sex.upper() == "M" else 1
    input_data = pd.DataFrame([{
        "age": payload.age,
        "sex_encoded": sex_encoded,
        "systolic_bp": payload.systolic_bp,
        "diastolic_bp": payload.diastolic_bp,
        "bmi": payload.bmi,
        "heart_rate": payload.heart_rate,
        "spo2": payload.spo2,
        "glucose": payload.random_glucose if payload.random_glucose is not None else float("nan")
    }])
    
    # 3. Model Inference
    probabilities = model.predict_proba(input_data)[0]
    predicted_idx = int(model.predict(input_data)[0])
    predicted_label = meta["labels"][predicted_idx]
    
    # 4. SHAP Explainability Extraction
    shap_values = explainer.shap_values(input_data)
    
    # Handle SHAP multi-class format depending on the library version
    if isinstance(shap_values, list):
        # Older SHAP: list of arrays
        class_shap = shap_values[predicted_idx][0]
    elif len(shap_values.shape) == 3:
        # Newer SHAP (v0.45+): 3D array (samples, features, classes)
        class_shap = shap_values[0, :, predicted_idx]
    else:
        class_shap = shap_values[0]
    
    evidence_for = []
    evidence_against = []
    
    for feature_name, shap_val, actual_val in zip(meta["features"], class_shap, input_data.iloc[0]):
        # Convert numeric categorical back to string for readability
        display_val = "M" if feature_name == "sex_encoded" and actual_val == 0 else ("F" if feature_name == "sex_encoded" and actual_val == 1 else actual_val)
        display_name = feature_name.replace("_", " ").title()
        
        reasoning = f"{display_name}: {display_val}"
        
        # We explicitly cast shap_val to float to prevent numpy array ambiguity
        shap_val_float = float(shap_val)
        
        if shap_val_float > 0.05: # Threshold for significant positive contribution
            evidence_for.append(reasoning)
        elif shap_val_float < -0.05: # Threshold for significant negative contribution
            evidence_against.append(reasoning)
            
    # 5. Rule Matrix Recommendations
    investigations = []
    if payload.systolic_bp >= 140 or payload.diastolic_bp >= 90:
        investigations.extend(["ECG (Resting)", "Lipid Profile", "Serum Creatinine"])
    if payload.random_glucose is not None and payload.random_glucose >= 140:
        investigations.extend(["HbA1c Test", "Fasting Blood Sugar"])

    return {
        "case_token": payload.case_token,
        "safety_screen_passed": True,
        "triage_urgency": "URGENT" if predicted_label == "high_risk" else ("MONITOR" if predicted_label == "moderate_risk" else "ROUTINE"),
        "differential_diagnosis": [
            {
                "condition_tier": predicted_label,
                "probability": round(float(probabilities[predicted_idx]), 3),
                "evidence_for": evidence_for,
                "evidence_against": evidence_against
            }
        ],
        "recommended_investigations": list(set(investigations)),
        "model_metadata": {
            "version": meta["model_version"],
            "calibrated": True
        }
    }

class ClinicalEvaluationRequest(BaseModel):
    case_token: str
    symptom_string: str
    age: int
    sex: str
    systolic_bp: float
    diastolic_bp: float
    bmi: float
    heart_rate: float
    random_glucose: Optional[float] = None
    spo2: float
    respiratory_rate: Optional[float] = None
    temperature: Optional[float] = None


@app.post("/api/v1/evaluate", response_model=KernelReportOutput)
async def evaluate_patient(payload: ClinicalEvaluationRequest):
    try:
        sex_encoded = 0 if payload.sex.upper() == "M" else 1

        refine_vitals = {
            "age": payload.age,
            "sex_encoded": sex_encoded,
            "systolic_bp": payload.systolic_bp,
            "diastolic_bp": payload.diastolic_bp,
            "bmi": payload.bmi,
            "heart_rate": payload.heart_rate,
            "spo2": payload.spo2,
            "glucose": payload.random_glucose if payload.random_glucose is not None else float("nan"),
        }

        triage_vitals = {
            "systolic_bp": payload.systolic_bp,
            "diastolic_bp": payload.diastolic_bp,
            "pulse": payload.heart_rate,
            "respiratory_rate": payload.respiratory_rate,
            "spo2": payload.spo2,
            "temperature": payload.temperature,
            "bmi": payload.bmi,
            "glucose": payload.random_glucose,
            "glucose_type": "fasting",
            "age": payload.age,
            "sex": payload.sex,
        }

        refinement_output = refine_diagnosis.refine(payload.symptom_string, refine_vitals)
        result = pipeline_glue.execute_full_clinical_pipeline(refinement_output, payload.age, triage_vitals)
        return KernelReportOutput(**result)
    except Exception:
        logger.exception("evaluate_patient failed (case_token=%s)", payload.case_token)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An internal error occurred while processing this evaluation.",
                "case_token": payload.case_token,
            },
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "samd-classifier",
        "model_version": meta.get("model_version", "unknown"),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
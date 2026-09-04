import argparse
import sys
import json
import requests

def test_api(base_url: str):
    print(f"\n=======================================================")
    print(f"🚀 Running Comprehensive SaMD Classifier API Tests")
    print(f"🎯 Target Base URL: {base_url}")
    print(f"=======================================================\n")

    passed = 0
    total = 0

    def record_result(name: str, success: bool, details: str = ""):
        nonlocal passed, total
        total += 1
        if success:
            passed += 1
            print(f"✅ PASS: {name}")
            if details:
                print(f"   {details}")
        else:
            print(f"❌ FAIL: {name}")
            if details:
                print(f"   {details}")

    # 1. Health Check
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        data = r.json()
        record_result(
            "GET /health",
            r.status_code == 200 and data.get("status") == "healthy",
            f"Status: {r.status_code}, Response: {data}"
        )
    except Exception as e:
        record_result("GET /health", False, f"Exception: {e}")

    # 2. POST /v1/assess - Borderline Patient
    patient_borderline = {
        "case_token": "case_borderline_01",
        "age": 35,
        "sex": "F",
        "systolic_bp": 138.0,
        "diastolic_bp": 82.0,
        "bmi": 31.7,
        "heart_rate": 77.0,
        "random_glucose": 110.0,
        "spo2": 99.0
    }
    try:
        r = requests.post(f"{base_url}/v1/assess", json=patient_borderline, timeout=10)
        data = r.json()
        success = (
            r.status_code == 200
            and data.get("safety_screen_passed") is True
            and len(data.get("differential_diagnosis", [])) > 0
        )
        diag = data.get("differential_diagnosis", [{}])[0]
        record_result(
            "POST /v1/assess (Borderline Case)",
            success,
            f"Triage: {data.get('triage_urgency')}, Predicted Tier: {diag.get('condition_tier')}, Prob: {diag.get('probability')}"
        )
    except Exception as e:
        record_result("POST /v1/assess (Borderline Case)", False, f"Exception: {e}")

    # 3. POST /v1/assess - Critical Emergency (SpO2 < 90, BP > 180)
    patient_emergency = {
        "case_token": "case_emergency_02",
        "age": 62,
        "sex": "M",
        "systolic_bp": 195.0,
        "diastolic_bp": 115.0,
        "bmi": 28.4,
        "heart_rate": 112.0,
        "random_glucose": 240.0,
        "spo2": 88.0
    }
    try:
        r = requests.post(f"{base_url}/v1/assess", json=patient_emergency, timeout=10)
        data = r.json()
        success = (
            r.status_code == 200
            and data.get("safety_screen_passed") is False
            and data.get("triage_urgency") == "EMERGENCY_REFERRAL"
        )
        record_result(
            "POST /v1/assess (Critical Emergency)",
            success,
            f"Triage Urgency: {data.get('triage_urgency')}, Safety Screen Passed: {data.get('safety_screen_passed')}"
        )
    except Exception as e:
        record_result("POST /v1/assess (Critical Emergency)", False, f"Exception: {e}")

    # 4. POST /api/v1/evaluate - Hypertension (I10) end-to-end evaluation
    hypertension_payload = {
        "case_token": "case_eval_i10_03",
        "symptom_string": "no symptoms | epistaxis | blurred vision",
        "age": 43,
        "sex": "M",
        "systolic_bp": 150,
        "diastolic_bp": 92,
        "bmi": 27.5,
        "heart_rate": 78,
        "random_glucose": 110,
        "spo2": 97,
        "respiratory_rate": 16,
        "temperature": 37.0
    }
    try:
        r = requests.post(f"{base_url}/api/v1/evaluate", json=hypertension_payload, timeout=15)
        data = r.json()
        diag_summary = data.get("diagnostic_summary", {})
        treatment = data.get("nlem_treatment", {})
        brand = data.get("brand_mapping", {})
        success = (
            r.status_code == 200
            and diag_summary.get("primary_icd_candidate") == "I10"
            and treatment.get("recommendedDrug") is not None
        )
        record_result(
            "POST /api/v1/evaluate (Hypertension I10 End-to-End)",
            success,
            f"Primary: {diag_summary.get('primary_icd_candidate')} ({diag_summary.get('primary_ailment_name')}), "
            f"Drug: {treatment.get('recommendedDrug')}, "
            f"Jan Aushadhi: {brand.get('jan_aushadhi_brand') if brand else None}"
        )
    except Exception as e:
        record_result("POST /api/v1/evaluate (Hypertension I10 End-to-End)", False, f"Exception: {e}")

    # 5. POST /api/v1/evaluate - General Symptoms (Headache)
    headache_payload = {
        "case_token": "case_eval_headache_04",
        "symptom_string": "headache",
        "age": 43,
        "sex": "M",
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "bmi": 22.0,
        "heart_rate": 75,
        "random_glucose": 90,
        "spo2": 98,
        "respiratory_rate": 16,
        "temperature": 37.0
    }
    try:
        r = requests.post(f"{base_url}/api/v1/evaluate", json=headache_payload, timeout=15)
        data = r.json()
        diag_summary = data.get("diagnostic_summary", {})
        success = (
            r.status_code == 200
            and len(diag_summary.get("differential", [])) > 0
        )
        record_result(
            "POST /api/v1/evaluate (Headache Clinical Evaluation)",
            success,
            f"Primary candidate: {diag_summary.get('primary_icd_candidate')}, Differentials count: {len(diag_summary.get('differential', []))}"
        )
    except Exception as e:
        record_result("POST /api/v1/evaluate (Headache Clinical Evaluation)", False, f"Exception: {e}")

    print(f"\n=======================================================")
    print(f"📊 Results: {passed}/{total} tests passed")
    print(f"=======================================================")

    return passed == total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SaMD Classifier API endpoints")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="Base URL of the service")
    args = parser.parse_args()

    success = test_api(args.url.rstrip("/"))
    sys.exit(0 if success else 1)

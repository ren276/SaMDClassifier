import os
import requests
import json

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/v1/assess")

# Test Case 1: Priya Sharma (From your Android Mockup)
# Borderline Grade 1 BP and High BMI
patient_1 = {
    "case_token": "sC3AkeU88dsA",
    "age": 35, 
    "sex": "F",
    "systolic_bp": 138.0,
    "diastolic_bp": 82.0,
    "bmi": 31.7,
    "heart_rate": 77.0,
    "random_glucose": 110.0,
    "spo2": 99.0
}

# Test Case 2: Critical Emergency 
# SpO2 drops below 90% and BP is in Hypertensive Crisis (>180)
patient_2 = {
    "case_token": "EMERG-999-X",
    "age": 62,
    "sex": "M",
    "systolic_bp": 195.0,
    "diastolic_bp": 115.0,
    "bmi": 28.4,
    "heart_rate": 112.0,
    "random_glucose": 240.0,
    "spo2": 88.0
}

def run_test(name, payload):
    print(f"==========================================")
    print(f"🧪 Testing: {name}")
    print(f"==========================================")
    try:
        # Sending the POST request to FastAPI
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            print("✅ SUCCESS! API Response:\n")
            # Parse and print the JSON response beautifully
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ FAILED with HTTP Status Code: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Connection refused.")
        print("Make sure your FastAPI server is running in another terminal tab using:")
        print("uvicorn src.app:app --reload")
    print("\n")

if __name__ == "__main__":
    print("Starting API Smoke Tests...\n")
    run_test("Patient 1 (Priya - Borderline)", patient_1)
    run_test("Patient 2 (Critical Emergency)", patient_2)
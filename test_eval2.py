import requests
import json

payload = {
    "case_token": "test_case",
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
    r = requests.post("http://127.0.0.1:8000/api/v1/evaluate", json=payload)
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("Error:", e)

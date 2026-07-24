import requests
import json

payload = {
    "case_token": "test_case",
    "symptom_string": "headache",
    "age": 43,
    "sex": "M",
    "systolic_bp": 128,
    "diastolic_bp": 82,
    "bmi": 29.4,
    "heart_rate": 78,
    "random_glucose": 210,
    "spo2": 98,
    "respiratory_rate": 16,
    "temperature": 98.6
}

try:
    r = requests.post("http://127.0.0.1:8000/api/v1/evaluate", json=payload)
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("Error:", e)

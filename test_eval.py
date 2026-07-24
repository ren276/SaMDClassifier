import requests
import json

payload = {
    "case_token": "test_case",
    "symptom_string": "no symptoms | epistaxis | blurred vision",
    "age": 43,
    "sex": "M",
    "systolic_bp": 150,
    "diastolic_bp": 92,
    "bmi": 27.5,
    "heart_rate": 78,
    "random_glucose": 110,
    "spo2": 97
}

try:
    r = requests.post("http://127.0.0.1:8000/api/v1/evaluate", json=payload)
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("Error:", e)

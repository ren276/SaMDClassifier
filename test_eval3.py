import requests
import json

payload = {
    "case_token": "test_case",
    "symptom_string": "headache",
    "age": 55,
    "sex": "F",
    "systolic_bp": 148,
    "diastolic_bp": 94,
    "bmi": 27.2,
    "heart_rate": 84,
    "random_glucose": 118,
    "spo2": 97,
    "respiratory_rate": 15,
    "temperature": 36.9
}

r = requests.post("http://127.0.0.1:8000/api/v1/evaluate", json=payload)
print(json.dumps(r.json(), indent=2))

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3500

# 1. Indian Demographic Distributions (NFHS-5 Baseline)
age = np.clip(np.random.normal(42, 14, N), 18, 85).astype(int)
sex = np.random.choice(["M", "F"], size=N, p=[0.51, 0.49])
sex_encoded = np.where(sex == "M", 0, 1)

# Asian Indian BMI Cutoffs (ICMR: >=23 Overweight, >=25 Obese)
bmi = np.clip(np.random.gamma(shape=6.5, scale=3.6, size=N), 14.5, 42.0)

# 2. Vitals Distribution (Calibrated to Indian Primary Care)
systolic_bp = np.clip(np.random.normal(126, 16, N) + (bmi - 22) * 0.8, 80, 210)
diastolic_bp = np.clip(82 + (systolic_bp - 126) * 0.45 + np.random.normal(0, 6, N), 50, 125)
heart_rate = np.clip(np.random.normal(78, 12, N), 48, 135)
random_glucose = np.clip(np.random.gamma(shape=5.0, scale=24, size=N), 60, 380)
spo2 = np.clip(100 - np.random.exponential(scale=2.2, size=N), 78, 100)

# 3. Target: High-Risk Hypertensive / Cardiometabolic Complication Score
# Using np.clip(array, min, max) for NumPy arrays
risk_score = (
    0.03 * np.clip(systolic_bp - 120, 0, None) +
    0.04 * np.clip(diastolic_bp - 80, 0, None) +
    0.02 * np.clip(age - 40, 0, None) +
    0.05 * np.clip(bmi - 23, 0, None) +
    0.02 * np.clip(random_glucose - 140, 0, None) +
    np.random.normal(0, 0.3, N)  # Realistic clinical noise
)

# Target Classes: 0: Low Risk, 1: Moderate Risk, 2: High Risk / Emergency Triage
target_risk = pd.qcut(risk_score, q=[0, 0.50, 0.82, 1.0], labels=[0, 1, 2])

df = pd.DataFrame({
    "age": age,
    "sex_encoded": sex_encoded,
    "systolic_bp": systolic_bp.round(1),
    "diastolic_bp": diastolic_bp.round(1),
    "bmi": bmi.round(1),
    "heart_rate": heart_rate.round(1),
    "random_glucose": random_glucose.round(1),
    "spo2": spo2.round(1),
    "risk_level": target_risk
})

df.to_csv("synthetic_patients.csv", index=False)
print("Synthetic Dataset Generated Successfully!")
print(df["risk_level"].value_counts(normalize=True) * 100)
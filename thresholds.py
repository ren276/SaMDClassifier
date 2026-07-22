"""Threshold helpers calibrated to the canonical synthetic dataset.

The thresholds are intended to mirror the generated dataset distribution and
the tiering used by the training pipeline, not external clinical guidelines.
"""


def classify_bp(
    systolic: float,
    diastolic: float,
    age: float | None = None,
    sex: str | None = None,
) -> str:
    if age is not None and age < 18:
        # Synthetic age-aware calibration for the canonical dataset.
        is_female = str(sex).upper().startswith("F") if sex is not None else False
        stage1_systolic = 110 + 1.6 * age + (1 if is_female else 0)
        stage1_diastolic = 79 + 0.7 * age + (1 if is_female else 0)
        if age >= 13:
            stage2_systolic = 140
            stage2_diastolic = 90
        else:
            stage2_systolic = stage1_systolic + 12
            stage2_diastolic = stage1_diastolic + 12

        if systolic >= stage2_systolic or diastolic >= stage2_diastolic:
            return "grade2"
        if systolic >= stage1_systolic or diastolic >= stage1_diastolic:
            return "grade1"
        return "normal"

    # Adult thresholds calibrated to the synthetic canonical dataset.
    if systolic >= 180 or diastolic >= 110:
        return "grade3"
    if systolic >= 160 or diastolic >= 100:
        return "grade2"
    if systolic >= 140 or diastolic >= 90:
        return "grade1"
    return "normal"
 

def classify_pulse(bpm: float) -> str:
    if bpm < 50:
        return "bradycardia"
    if bpm < 60:
        return "low-normal-pulse"
    if bpm <= 100:
        return "normal-pulse"
    if bpm <= 130:
        return "tachycardia"
    return "severe-tachycardia"


def classify_respiratory_rate(breaths_per_min: float) -> str:
    if breaths_per_min < 10:
        return "severe-bradypnea"
    if breaths_per_min < 12:
        return "bradypnea"
    if breaths_per_min <= 18:
        return "normal-resp-rate"
    if breaths_per_min <= 25:
        return "mild-tachypnea"
    return "severe-tachypnea"


def classify_spo2(percent: float) -> str:

    if percent < 90:
        return "severe-hypoxemia"
    if percent < 95:
        return "hypoxemia"
    return "normal-spo2"


def classify_temperature(celsius: float) -> str:
    if celsius < 35:
        return "hypothermia"
    if celsius < 36.1:
        return "low-normal-temp"
    if celsius <= 38.0:
        return "normal-temp"
    if celsius <= 40.0:
        return "fever"
    return "hyperpyrexia"


def classify_glucose(mg_dl: float, reading_type: str = "fasting") -> str:
    if reading_type != "fasting":
        return "glucose-reading-type-unconfirmed"
    if mg_dl < 70:
        return "hypoglycemia"
    if mg_dl < 100:
        return "normal-glucose"
    if mg_dl < 126:
        return "prediabetic-range"
    return "diabetic-range"


def classify_bmi(bmi: float) -> str:

    if bmi < 18.5:
        return "underweight"
    if bmi < 23:
        return "normal-bmi"
    if bmi < 25:
        return "overweight"
    return "obese"


def combine_vitals(
    systolic_bp: float, diastolic_bp: float, pulse: float, respiratory_rate: float,
    spo2: float, temperature: float, bmi: float, glucose: float | None = None,
    glucose_type: str = "fasting",
    age: float | None = None,
    sex: str | None = None,
) -> dict:
    flags = {
        "bp_grade": classify_bp(systolic_bp, diastolic_bp, age=age, sex=sex),
        "pulse": classify_pulse(pulse),
        "respiratory_rate": classify_respiratory_rate(respiratory_rate),
        "spo2": classify_spo2(spo2),
        "temperature": classify_temperature(temperature),
        "bmi": classify_bmi(bmi),
    }
    if glucose is not None:
        flags["glucose"] = classify_glucose(glucose, glucose_type)

    urgent_flags = {
        "grade3", "grade2", "severe-tachycardia", "severe-bradypnea", "severe-tachypnea",
        "severe-hypoxemia", "hyperpyrexia", "hypothermia", "hypoglycemia", "diabetic-range",
    }
    moderate_flags = {
        "grade1", "tachycardia", "bradycardia", "mild-tachypnea", "bradypnea",
        "hypoxemia", "fever", "prediabetic-range", "obese",
    }
    flag_values = set(flags.values())
    if flag_values & urgent_flags:
        overall = "urgent-review"
    elif flag_values & moderate_flags:
        overall = "monitor"
    else:
        overall = "routine"

    flags["overall_urgency"] = overall
    return flags


if __name__ == "__main__":
    cases = [
        dict(age=10, sex="F", systolic_bp=118, diastolic_bp=76, pulse=72, respiratory_rate=14,
             spo2=98, temperature=36.8, bmi=22.0, glucose=88, glucose_type="fasting"),
        dict(age=35, sex="F", systolic_bp=150, diastolic_bp=95, pulse=110, respiratory_rate=22,
             spo2=94, temperature=39.1, bmi=31.7, glucose=140, glucose_type="fasting"),
        dict(age=62, sex="M", systolic_bp=185, diastolic_bp=100, pulse=130, respiratory_rate=28,
             spo2=88, temperature=40.5, bmi=28.0, glucose=210, glucose_type="fasting"),
    ]
    for c in cases:
        print(c)
        print(combine_vitals(**c))
        print()
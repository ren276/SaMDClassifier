# DRISHTI/PHC — Foundational Synthetic Vitals Dataset

> **⚠️ SYNTHETIC DATA — NOT REAL PATIENT DATA**
> All records in this dataset are 100% synthetically generated. `synthetic=True`
> is a mandatory column on every row. This flag must be preserved on all downstream
> joins, merges, and data pipeline stages. Do not conflate this data with real
> patient data at any stage.

---

## Purpose

This dataset is the **foundation/bootstrap layer** for the DRISHTI/PHC SaMD offline
AI correlation kernel (IIT Indore, Prof. Banda, HarSaRK project). It is used to
initialize model training until CDSCO/government API access to real patient data
is granted, at which point the model will be retrained from scratch on real data.

**The pipeline produces a single canonical CSV file** — not multiple per-disease
files. All records share one feature space (5 vital signs). This is required for
Layer 1 to train as a single multi-label model over a shared taxonomy.

---

## Dataset Properties

| Property | Value |
|---|---|
| Generation method | Synthea (MITRE) with India `biometrics.yml` override |
| Target row count | 20,000 |
| Age range | 5–80 years |
| Records per age group | Adults (≥18): IHCI/WHO-Asian thresholds; Pediatric (5–17): Narang et al. BP formula |
| Pediatric BMI | **Not assessed this pass** — see Deferred Items below |
| All records | `synthetic=True`, `source="synthea_india"` |
| Reproducibility | Seed list + `noise_log.json` |

---

## Clinical Threshold Sources

> *"Vitals thresholds are India-calibrated (IHCI/ICMR/WHO-Asian). All demographic,
> geographic, provider, and payer fields from Synthea's base generator are
> discarded — not used, not representative of India, present only as generator
> scaffolding."*

### Adult Thresholds (age ≥ 18)

| Vital | Threshold | Source |
|---|---|---|
| BP Systolic HTN | ≥ 140 mmHg (Stage 1) | IHCI 2019 — ihci.in |
| BP Diastolic HTN | ≥ 90 mmHg (Stage 1) | IHCI 2019 |
| SpO2 Abnormal | < 95% | Standard clinical |
| BMI Overweight | ≥ 23 kg/m² | WHO Asian cutoffs, ICMR-adopted |
| BMI Obese | ≥ 25 kg/m² | WHO Asian cutoffs |
| Glucose Diabetes | ≥ 126 mg/dL | ICMR-INDIAB |
| Pulse Tachycardia | > 100 bpm | Universal |
| Pulse Bradycardia | < 60 bpm | Universal |

**Note:** AHA 2017 ≥130 mmHg SBP cutoff is explicitly NOT used. IHCI ≥140 is the source.

### Pediatric Thresholds (age 5–17)

| Vital | Formula/Rule | Source |
|---|---|---|
| SBP Hypertension (95th pct) | `110 + 1.6 × age` mmHg (+1 for female) | Narang et al., AIIMS, *Indian Pediatrics* |
| DBP Hypertension (95th pct) | `79 + 0.7 × age` mmHg (+1 for female) | Narang et al., AIIMS, *Indian Pediatrics* |
| BP Stage 1 | 95th pct to 95th+12 mmHg | IAP/AAP-aligned, Indian rural-BP study |
| BP Stage 2 | Above Stage 1 upper | IAP/AAP-aligned |
| BMI | **Not assessed** — see Deferred Items | IAP 2015 growth charts (deferred) |

Anchor values: ~120/80 at age 5, ~125/85 at age 10, ~135/90 at age 15.

---

## Label Architecture

Labels are **NOT** from Synthea's internal disease modules. They are derived
from `abnormal_params` combinations via the mapping table in `config.py`.

The label hierarchy is three-level:

```
icd_chapter (always populated — system level)
  └── icd_block (always populated — category/block range)
        └── icd_candidate (nullable — specific code, only where clinically warranted)
```

`icd_candidate` is intentionally null for combinations that do not plausibly
narrow below block level. Null rows are **not errors** — they are the rows the
LLM/retrieval Layer 2 is designed to handle. Do not fabricate icd_candidate
values to fill nulls.

`symptom_signal_strength` (`strong` / `supportive` / `nonspecific`) encodes
the confidence-basis training signal for Layer 1. This feeds the physician-
verification banner logic. It is a distinct, independently queryable column
for ISO 14971 / SaMD risk documentation.

---

## Schema

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `patient_id` | str | No | UUID |
| `encounter_date` | date | No | ISO-8601 |
| `age_at_encounter` | int | No | years |
| `sex` | str | No | M / F |
| `bp_systolic` | float | No | mmHg |
| `bp_diastolic` | float | No | mmHg |
| `pulse` | float | No | bpm |
| `spo2` | float | No | % |
| `bmi` | float | Yes | kg/m² — may be NaN for some encounters |
| `tier` | int | No | 1–4 |
| `abnormal_params` | str | No | comma-sep, alpha-sorted |
| `symptom_string` | str | No | sampled from pool per mapping table |
| `symptom_signal_strength` | str | No | `strong` / `supportive` / `nonspecific` |
| `icd_chapter` | str | No | ICD-10 chapter name |
| `icd_block` | str | No | ICD-10 block range |
| `icd_candidate` | str | **Yes** | Specific ICD-10 code or null |
| `differential_candidates` | str | No | pipe-sep ranked list (block granularity) |
| `synthetic` | bool | No | Always `True` |
| `source` | str | No | Always `"synthea_india"` |

---

## Deferred Items (Future Labeling-Only Re-Run)

### Pediatric BMI (IAP 2015 Growth Charts)

IAP defines overweight as BMI ≥ 23-adult-equivalent percentile and obese as
≥ 27-adult-equivalent (age+sex chart). This requires a lookup table, not a formula.

**Action when ready:**
1. Build `drishti_pipeline/data/iap_bmi_percentiles.csv` (age 5–17, M/F, cutoff per row)
2. Set `PEDIATRIC_BMI_STRATEGY = "B1"` in `config.py`
3. Re-run `step3_tiered_generator.py` on the existing canonical data
4. Re-run `step4_symptom_pairing.py`, `step5_aggregate.py`, `step6_noise_injection.py`

**No Synthea regeneration needed.** The vitals_raw files in `scratch/` are preserved.

> ⚠️ This re-run must be completed before any BMI-specific pediatric model training.

---

## Reproducibility

The pipeline is fully reproducible given:
- The seed list used across pipeline passes (printed during `run_pipeline.py` execution)
- `noise_log.json` (noise seed + parameters)
- `biometrics.yml` override (committed in `synthea-international/in/`)
- `config.py` (threshold constants + mapping table)

The `canonical_dataset_prenoise.csv` preserves pre-noise values for audit purposes.

---

## Contact

Sandesh / IIT Indore / Prof. Banda / HarSaRK project  
DRISHTI/PHC SaMD — offline AI correlation kernel  
Foundation dataset — to be retrained on real data upon CDSCO/government API access

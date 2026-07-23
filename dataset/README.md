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
files. All records share one feature space (6 vital signs). This is required for
Layer 1 to train as a single multi-label model over a shared taxonomy.

---

## Dataset Properties

| Property | Value |
|---|---|
| Generation method | Synthea (MITRE) with India `biometrics.yml` override + post-hoc demographic reweighting |
| Target row count | ~27,000 (configurable via `run_pipeline.py --target-rows`) |
| Age range | 5–80 years |
| Records per age group | Adults (≥18): IHCI/WHO-Asian/ICMR-INDIAB thresholds; Pediatric (5–17): Narang et al. BP formula |
| Pediatric BMI | **Not assessed this pass** — see Deferred Items below |
| All records | `synthetic=True`, `source="synthea_india"` |
| Reproducibility | Seed list + `noise_log.json` |

---

## Clinical Threshold Sources

> *"Vitals thresholds are India-calibrated (IHCI/ICMR/WHO-Asian). All demographic,
> geographic, provider, and payer fields from Synthea's base generator are
> discarded — not used, not representative of India, present only as generator
> scaffolding (age/sex distribution is corrected separately, see Demographics
> Reweighting below)."*

### Adult Thresholds (age ≥ 18)

| Vital | Threshold | Source |
|---|---|---|
| BP Systolic HTN | ≥ 140 mmHg (Stage 1) | IHCI 2019 — ihci.in |
| BP Diastolic HTN | ≥ 90 mmHg (Stage 1) | IHCI 2019 |
| SpO2 Abnormal | < 95% | Standard clinical |
| BMI Overweight | ≥ 23 kg/m² | WHO Asian cutoffs, ICMR-adopted |
| BMI Obese | ≥ 25 kg/m² | WHO Asian cutoffs |
| Glucose Abnormal (random/non-fasting) | ≥ 140 mg/dL (prediabetes), ≥ 200 mg/dL (diabetes) | RSSDI / ICMR-INDIAB random-glucose screening criterion |
| Pulse Tachycardia | > 100 bpm | Universal |
| Pulse Bradycardia | < 60 bpm | Universal |

**Note:** AHA 2017 ≥130 mmHg SBP cutoff is explicitly NOT used. IHCI ≥140 is the source.

**Note on glucose — two distinct numbers, do not conflate them:**
1. `synthea-international/in/.../biometrics.yml`'s `glucose: [70, 100, 145, 220]` bins
   control what raw glucose *value* Synthea draws for a patient it has already
   internally tagged as normoglycemic/prediabetic/diabetic (via its own
   US-epidemiology-driven `diabetes_severity` disease module — out of scope to
   edit). These bins were widened from an earlier `[70, 100, 126, 200]`
   (fasting-style) because a fasting-shaped bin under a random-glucose
   classification threshold caused `glucose_high` to almost never realize
   (~1.1% of draws ≥126, 0% ≥200, measured pre-fix).
2. `config.ADULT_THRESHOLDS["glucose"]` (140/200 above) is the *classification*
   threshold this pipeline applies to whatever value gets drawn, per the
   RSSDI/ICMR-INDIAB random-glucose screening criterion. This is intentionally
   different from any *fasting* glucose criterion (100/126) — the two should
   never be set to the same numbers, since random and fasting glucose have
   different clinical cutoffs.

### Pediatric Thresholds (age 5–17)

| Vital | Formula/Rule | Source |
|---|---|---|
| SBP Hypertension (95th pct) | `110 + 1.6 × age` mmHg (+1 for female) | Narang et al., AIIMS, *Indian Pediatrics* |
| DBP Hypertension (95th pct) | `79 + 0.7 × age` mmHg (+1 for female) | Narang et al., AIIMS, *Indian Pediatrics* |
| BP Stage 1 | 95th pct to 95th+12 mmHg | IAP/AAP-aligned, Indian rural-BP study |
| BP Stage 2 | Above Stage 1 upper | IAP/AAP-aligned |
| BMI | **Not assessed** — see Deferred Items | IAP 2015 growth charts (deferred) |
| Glucose | Same 140/200 mg/dL random-glucose cutoff as adults, no age-gating | RSSDI / ICMR-INDIAB |

Anchor values: ~120/80 at age 5, ~125/85 at age 10, ~135/90 at age 15.

---

## Label Architecture

Labels are **NOT** from Synthea's internal disease modules. They are derived
from `abnormal_params` combinations via the mapping table in `config.py`
(`ICD_MAPPING`), resolved through `config.resolve_condition()`.

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

### Weighted multi-candidate resolution (Rev 6)

Several conditions (anemia, hyper/hypothyroidism, anxiety/panic, migraine,
osteoarthritis, and the vector-borne/PHC-burden infections) have no
clinically-distinguishable vital-sign fingerprint of their own in a 6-signal
system — a raised pulse alone can't mechanically distinguish sinus
tachycardia from anxiety from anemia from hyperthyroidism. Rather than force
each into an artificial unique vital combination, `ICD_MAPPING` entries for
these keys carry a `candidates` list instead of a single flat condition, each
with a `weight` (and optionally a `monsoon_weight` override for June-September).

`config.resolve_condition(abnormal_params_set, patient_id, encounter_date)`
deterministically picks one candidate per row via a stable SHA-256 hash of
`(patient_id, encounter_date, salt)` — same inputs always resolve to the same
condition. **Both `step3` (icd_chapter/block/candidate labeling) and `step4`
(symptom + drug/dosage sampling) call this exact same resolver** so a row's
label and its sampled symptoms/drug can never silently disagree (e.g. row
labeled "dengue" while symptoms/drug were sampled for "malaria").

### `fever_pattern` — a synthetic signal, not a measured vital

A 6-vital system (2×BP, pulse, SpO2, BMI, glucose) has no temperature/CBC/
serology channel to carry infectious-disease signal. Without one,
malaria/dengue/typhoid/chikungunya/UTI/gastroenteritis/TB/LRTI would be
permanently stuck as text inside `differential_candidates` and never realize
as a primary `icd_candidate` (this is exactly what happened before Rev 6 — a
monsoon-season 60%-probability promotion existed in code but its preconditions
compounded to near-zero and it never actually fired in the generated data).

`fever_pattern_flag` is a deterministic, hash-based flag (`config.
assess_fever_pattern`) standing in for "this encounter presents with a
fever/infectious pattern" — base rate ~8% year-round, ~22% during monsoon
(Jun-Sep). It is appended to `abnormal_params` for ICD/symptom resolution
purposes but **does NOT count toward `tier`** — tier measures physiological
vital-sign derangement severity, not epidemiological routing.

---

## Demographics Reweighting (Rev 6)

Synthea's built-in demographic generator draws age/sex from a US-pattern
distribution; the India `biometrics.yml` override only recalibrates vital/lab
*ranges* conditioned on age/sex, not the age/sex draw itself, and no India
demographics/geography module exists in `synthea-international/in` (unlike
`de`, `fr`, `gb`, etc., which ship one).

`drishti_pipeline/step2b_demographic_reweight.py` corrects this post-hoc: it
accumulates a raw pool several times larger than the target row count
(`config.RAW_POOL_MULTIPLIER`), then resamples per age-decade × sex cell to
match `config.TARGET_AGE_SEX_DISTRIBUTION` (approximate Census 2011 broad
age-band shares, renormalized to this pipeline's 5–80 window) and
`config.TARGET_SEX_SHARE` (~51.5% M / 48.5% F, Census 2011 sex ratio).
Precision is not required for a synthetic bootstrap dataset — this is a
documented approximation, not a demographic model. Under-represented cells
(typically the elderly bands) are resampled **with replacement**; the
reweighting report printed during the run flags which cells needed this.

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
| `bmi` | float | Yes | kg/m² — opportunistic, may be NaN |
| `glucose` | float | Yes | mg/dL, random/non-fasting — opportunistic, may be NaN (sparser than bmi; deliberately NOT imputed) |
| `bp_systolic_observed` … `glucose_observed` | float | Yes (mirrors base column) | Gaussian-noised "as measured" counterparts (see `config.NOISE_PARAMS`); labels are based on the pre-noise columns above, never recalculated post-noise |
| `tier` | int | No | 1–6 (count of abnormal vital-sign params; `fever_pattern` excluded from this count). Tiers 4–6 are naturally rare (5-6 simultaneous abnormal vitals measured at ~0% even in a 450k-row pool) — `config.GLOBAL_TIER_TARGET_RATIOS` are soft upper-bound caps, not guarantees, and this is accepted as clinically realistic rather than corrected for. |
| `abnormal_params` | str | No | comma-sep, alpha-sorted; may include `fever_pattern` |
| `fever_pattern_flag` | bool | No | Synthetic epidemiological routing signal, NOT a measured vital — see Label Architecture |
| `symptom_string` | str | No | sampled from the resolved condition's symptom pool |
| `symptom_signal_strength` | str | No | `strong` / `supportive` / `nonspecific` |
| `drug_name` | str | No | Sampled from the resolved condition's prescription pool; `"None"` for age < 18 (hard-enforced, see Pediatric Safety below) |
| `drug_dosage` | str | No | Paired with `drug_name`; `"None"` for age < 18 |
| `icd_chapter` | str | No | ICD-10 chapter name |
| `icd_block` | str | No | ICD-10 block range |
| `icd_candidate` | str | **Yes** | Specific ICD-10 code or null |
| `differential_candidates` | str | No | pipe-sep ranked list (block granularity) |
| `pediatric_referral_flag` | bool | No | `True` for age < 18 — always paired with `drug_name="None"` |
| `synthetic` | bool | No | Always `True` |
| `source` | str | No | Always `"synthea_india"` |

### Pediatric Safety (hard-enforced invariant)

Every row with `age_at_encounter < 18` gets `drug_name="None"`,
`drug_dosage="None"`, `pediatric_referral_flag=True` — regardless of which
condition was resolved. This is a blanket override in `step4`, not a per-
condition rule, so it can't be silently bypassed by adding new `ICD_MAPPING`
entries. `step5_aggregate.py::run()` and `run_pipeline.py::run_spot_checks()`
both independently hard-fail (`AssertionError`) if any under-18 row has a
non-null drug — this exact invariant regressed once already in this
project's history (drug/dosage/referral columns were dropped from a dataset
rebuild that only re-ran the noise-injection step), so it is checked at two
separate points on every run, not just once.

---

## Disease Coverage

~19 distinct conditions across the four abnormal-vital-count tiers plus the
`fever_pattern` axis — see `metadata.json`'s `disease_coverage` object for the
full `condition_tag` → `icd_candidate` list and which resolve via the
weighted multi-candidate mechanism vs. a flat unique key (ISO 14971
traceability). Includes: essential hypertension (stages 1–2), obesity,
osteoarthritis, migraine, white-coat hypertension, sinus tachycardia/
bradycardia, anxiety/panic disorder, anemia, hyper/hypothyroidism, type-2
diabetes mellitus, metabolic syndrome, malaria, dengue fever (+ warning
signs / hemorrhagic), typhoid, chikungunya, urinary tract infection, acute
gastroenteritis, pulmonary tuberculosis, lower respiratory tract infection,
and severe pneumonia with sepsis.

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
- `config.py` (threshold constants + mapping table + demographic targets)
- The demographic reweighting seed (printed during `step2b` execution)

The `canonical_dataset_prenoise.csv` preserves pre-noise values for audit purposes.

**Rev 6 note:** the pipeline moved from an incremental per-Synthea-seed append
model to a one-batch-run model (accumulate → reweight → global tier/label
once — see `run_pipeline.py`). Re-running with different code no longer
appends to an existing `canonical_dataset.csv`; regenerate from a cleared
`drishti_dataset/` + `drishti_pipeline/scratch/` for a clean baseline.

---

## Contact

Sandesh / IIT Indore / Prof. Banda / HarSaRK project  
DRISHTI/PHC SaMD — offline AI correlation kernel  
Foundation dataset — to be retrained on real data upon CDSCO/government API access

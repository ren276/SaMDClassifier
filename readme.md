# samd-classifier

Toy ML classifier service for the PHC Patient Care SaMD app (`SaMD-App`). Standalone Python
service, called over HTTP by the Android app's Consultation flow. Not part of the Android build.

Current scope: **hypertension risk stratification only** (single disease). Designed to expand to
additional diseases later without changing the API contract shape — see
`docs/api-contracts/classifier-v1.md`.

## Status

| Component | Status |
|---|---|
| Synthetic dataset threshold table (`thresholds.py`) | done |
| Synthetic dataset generator | done |
| XGBoost training pipeline | done |
| FastAPI service (`/v1/assess`) | in progress |
| SHAP-based reasoning strings | in progress |
| Smoke tests (`test_api.py`) | not started |
| Real (non-synthetic) training data | not started — blocked on data source |
| Second disease (diabetes risk) | not started — deferred until v1 is stable |
| Encryption envelope for external/govt API calls | not started — deferred, see contract doc |

## Why this is a separate repo from SaMD-App

This is a separately deployed service with its own dependencies (Python/ML stack vs.
Kotlin/Android), its own release cadence, and no shared build pipeline with the app. The Android
app talks to it only over the HTTP contract in `docs/api-contracts/`. Treat that contract file as
the single source of truth both repos code against — do not let either side drift from it without
a version bump.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt --break-system-packages
```

## Pipeline, in order

```bash
./sync_dataset.sh [drishti_dataset_dir]  # pull the latest canonical dataset from the
                                          # upstream drishti_pipeline output (default source
                                          # dir is machine-local, see script header; override
                                          # via $1 or $DRISHTI_DATASET_DIR). Run this whenever
                                          # the upstream pipeline has regenerated the dataset —
                                          # do NOT train against a stale local copy.
python thresholds.py                  # sanity-check the gold-standard boundaries
    # writes synthetic_patients.csv
python train_model.py                 # writes model.json + model_meta.json, appends training_log.jsonl
python train_symptom_classifier.py    # writes symptom_model.json + symptom_model_meta.json,
                                       # symptom_vectorizer_{word,char}.joblib,
                                       # symptom_label_encoder.joblib, appends
                                       # symptom_training_log.jsonl
uvicorn app:app --reload              # serves POST /v1/assess on localhost:8000
python test_api.py                    # smoke test against the running server
```

## Model versioning

`model_meta.json` (written by `train_model.py`) is the source of truth for what's currently
deployed: `model_version`, `training_data_source`, `features`, `labels`, evaluation metrics,
training date. Bump `model_version` (e.g. `toy-v0.1` → `toy-v0.2`) on any retrain that changes
features, thresholds, or training data — never silently overwrite a version in place.

`training_log.jsonl` is the append-only run history (one JSON line per `train_model.py` run,
including timestamp, dataset source, hyperparameters, and metrics) — unlike `model_meta.json`,
which is overwritten each run to reflect only the currently-deployed model, this file is never
truncated, so past runs stay auditable.

## Training data details

- **Features (8):** `age, sex_encoded, systolic_bp, diastolic_bp, bmi, heart_rate, spo2,
  glucose`. The six vitals (`systolic_bp, diastolic_bp, bmi, heart_rate, spo2, glucose`) are
  sourced from the dataset's noised `_observed` columns (as-measured values), not the
  ground-truth columns — that's the point of the pipeline's noise-injection step. `age` and
  `sex_encoded` have no `_observed` variant and come from the ground-truth columns.
- **Glucose sparsity:** `glucose`/`glucose_observed` are intentionally sparse (~35% notna —
  opportunistic screening, not every encounter gets a glucose test). Rows with missing glucose
  are **not** dropped and the value is **not** imputed — it's left as `NaN` and handled by
  XGBoost's native missing-value split routing. Dropping rows would bias training toward
  whichever subpopulation happened to get tested (which likely correlates with acuity, i.e.
  the label itself).
- **Target mapping:** dataset `tier` (1–4, count of abnormal vital-sign params, tier 4 =
  "3+ abnormal params") collapses to the API's 3-class `low_risk/moderate_risk/high_risk` via
  `{1: low, 2: moderate, 3: high, 4: high}`. Tiers 3 (~6.5% of rows) and 4 (~1.5%) are merged
  into `high_risk` because tier 4 alone is too small a sample for a stable per-class F1.

## Classifier B — symptom-to-disease (`train_symptom_classifier.py`)

Separate model lineage from Classifier A (tier risk). Predicts `icd_candidate` (disease code)
from free-text `symptom_string`, TF-IDF + XGBoost, same model family as Classifier A for
architectural/SHAP consistency. Version lineage: `symptom-clf-vX-...` (not `toy-vX`, since it's
a different model, not a Classifier A retrain).

- **Row filter:** trained only on rows with non-null `icd_candidate` (~68% of the dataset —
  the rest are tier-1/no-disease-signal rows by dataset design, not missing data).
- **Features:** word-level TF-IDF (1,2-gram) + char-level TF-IDF (`char_wb`, 3-5 gram),
  combined. Both vectorizers are persisted (`symptom_vectorizer_word.joblib`,
  `symptom_vectorizer_char.joblib`) — required to vectorize new text at inference time; the
  model file alone is not usable without them.
- **Output:** `predict_ranked()` in the training script returns the full class list sorted by
  probability, not just the top prediction — this is the shape a future refinement/re-ranking
  stage (not yet built) will consume.
- **Known limitation — training data has no spelling/phrasing variance.** `symptom_string` is
  sampled from a small fixed per-condition phrase pool (dataset generation, not real patient
  entry). Verified directly: individual symptom phrases *are* shared across multiple conditions
  (e.g. "fatigue" appears under `E66`, `E11`, and `D50`), but each condition's *3-phrase
  combination* pool is disjoint from every other condition's, so the training/test data is
  perfectly linearly separable by TF-IDF (test-set accuracy and per-class F1 are 1.0000 across
  all 18 classes as of `symptom-clf-v0.1-tfidf-xgboost`). **This is a real property of the
  synthetic dataset, not a training bug** — but it means these metrics say nothing about
  real-world performance against actual free-text clinical entry, which will have typos,
  synonyms, incomplete phrasing, and out-of-vocabulary terms this dataset never generates. The
  char-ngram vectorizer is a partial mitigation for misspellings, not a validated one — there is
  no noisy/OOV eval set yet. Treat this model as unvalidated for real input until tested against
  non-synthetic or deliberately-perturbed symptom text.

## Stage 2 — refinement/re-ranking (`refine_diagnosis.py`)

Combines Classifier A's vitals-based risk tier with Classifier B's symptom-based ranked
differential into one recalibrated ranked list (top 3-5, never a single winner) for physician
review. Not wired into `app.py` yet.

- **Method chosen: weighted blend (not a secondary ML model, not discrete if/else rules).**
  Each `icd_candidate`'s symptom-model probability is multiplied by a `vitals_tier_alignment`
  score — the overlap between that candidate's empirical tier profile (read from the training
  dataset) and what Classifier A predicted for this patient's vitals — then renormalized.
  Rejected a secondary XGBoost meta-model: several `icd_candidate` classes are already thin
  (`B50`=21, `J22`=30, `A91`=50 rows), so stacking another trained model on top adds variance
  and an opaque layer for marginal gain, against this project's existing no-black-box posture
  (SHAP-explainable Classifier A, no ML-generated dosing). The blend is the more general,
  equally auditable form of the same "penalize vitals-implausible candidates" rule-based idea.
- **Bounded, never zeroes a candidate:** multiplier is `0.5 + alignment` (range 0.5–1.5), so no
  candidate is ever algorithmically eliminated from the ranking, only re-weighted.
- **Caveat — same "too-clean synthetic data" pattern as Classifier B itself:** most
  `icd_candidate` classes in this dataset are ~100% concentrated at a single tier (verified via
  `icd_candidate` × `tier` co-occurrence), making `vitals_tier_alignment` close to categorical
  (0.0 or 1.0) for most classes. It comes out genuinely soft (e.g. 0.41–0.94) only for the
  handful of classes whose real tier profile is mixed (`A91`, `I10`, `J22`) — confirmed by
  running the demo cases: alignment scores for these were 0.59–0.94, never collapsing to a flat
  0/1, so the formula does distinguish genuine ambiguity from the categorical majority rather
  than treating everything as certain. (Where the *final* adjusted confidence still came out
  high on an ambiguous case, e.g. I10 at 99.3%, that traces back to Classifier B's own
  already-high symptom-based confidence on that specific input — the alignment step only mildly
  adjusted it — not to the re-ranking formula manufacturing false certainty.)
- **`DiagnosisFeedback`** (in `refine_diagnosis.py`) defines the physician-feedback data
  contract for a future capture step: `physician_decision` reuses the Android app's exact
  `KernelDecision` values (`AGREE` / `MODIFY` / `REJECT`). **Schema only this pass** — no
  capture endpoint, no persistence, no export/re-import into `drishti_pipeline` is built yet.
  When that export step is built: only `AGREE`/`MODIFY` cases are eligible for dataset
  re-import (confirmed diagnoses); `REJECT` and any unconfirmed case must never be reimported —
  there's no trustworthy ground truth in either case.

## Regulatory notes (working, not final)

- Thresholds in `thresholds.py` are calibrated to the canonical synthetic dataset in
  `dataset/` and the tier structure used for training. They are not sourced from external
  clinical guidelines.
- Training data is synthetic (`synthetic-v1`) — not yet validated against real patient data or
  aggregated literature distributions. Track this explicitly; do not present synthetic-trained
  metrics as clinically validated performance.
- No patient-identifying fields are used as model features (`patient_id` travels in the request
  for audit correlation only — see contract doc).

## Docs

- `docs/api-contracts/classifier-v1.md` — frozen request/response contract
- `docs/api-contracts/classifier-v1.examples.json` — shared fixtures for tests on both sides
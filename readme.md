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
python thresholds.py                  # sanity-check the gold-standard boundaries
    # writes synthetic_patients.csv
python train_model.py                 # writes model.json + model_meta.json
uvicorn app:app --reload              # serves POST /v1/assess on localhost:8000
python test_api.py                    # smoke test against the running server
```

## Model versioning

`model_meta.json` (written by `train_model.py`) is the source of truth for what's currently
deployed: `model_version`, `training_data_source`, `features`, `labels`, evaluation metrics,
training date. Bump `model_version` (e.g. `toy-v0.1` → `toy-v0.2`) on any retrain that changes
features, thresholds, or training data — never silently overwrite a version in place.

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
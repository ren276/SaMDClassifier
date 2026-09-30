"""
Wire-contract guards for model identity and the worker-safe explanation. Run directly:
python -m unittest discover -s tests

/evaluate stubs only the NLEM retrieval step (a sentence-transformers download in a fresh
environment); the models, the ranking and the response schema are real.
"""

import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fastapi.testclient import TestClient

import app as app_module
import pipeline_glue
from rag.embedding_pin import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION

ASSESS = {
    "case_token": "case_identity",
    "age": 35,
    "sex": "F",
    "systolic_bp": 120.0,
    "diastolic_bp": 80.0,
    "bmi": 22.0,
    "heart_rate": 75.0,
    "random_glucose": 90.0,
    "spo2": 98.0,
}
EMERGENCY = {**ASSESS, "systolic_bp": 195.0, "diastolic_bp": 115.0, "spo2": 88.0}
EVALUATE = {
    **ASSESS,
    "symptom_string": "cough and fever for three days with chest pain",
    "respiratory_rate": 18.0,
    "temperature": 37.0,
}
NO_RETRIEVAL = {"requiresHumanReview": False, "pediatric_referral_flag": False}


class ModelIdentity(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)

    def _evaluate(self):
        with mock.patch.object(pipeline_glue, "get_treatment_recommendation", return_value=NO_RETRIEVAL):
            response = self.client.post("/api/v1/evaluate", json=EVALUATE)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_assess_metadata_has_every_key_and_version_equals_model_version(self):
        for payload in (ASSESS, EMERGENCY):  # the red-flag branch builds it separately
            metadata = self.client.post("/v1/assess", json=payload).json()["model_metadata"]
            self.assertEqual(set(metadata), {"model_version", "version", "model_sha256", "calibrated"})
            self.assertEqual(metadata["version"], metadata["model_version"])
            self.assertRegex(metadata["model_sha256"], r"^[0-9a-f]{64}$")

    def test_assess_sha256_is_the_verified_model_file_hash(self):
        metadata = self.client.post("/v1/assess", json=ASSESS).json()["model_metadata"]
        self.assertEqual(metadata["model_sha256"], app_module.VERIFIED_ARTIFACTS["models/model.json"])

    def test_evaluate_carries_model_metadata(self):
        metadata = self._evaluate()["model_metadata"]
        self.assertEqual(
            set(metadata),
            {
                "vitals_model_version",
                "vitals_model_sha256",
                "symptom_model_version",
                "symptom_model_sha256",
                "calibrated",
                "embedding_model",
            },
        )
        self.assertEqual(set(metadata["calibrated"]), {"vitals", "symptom"})
        self.assertEqual(metadata["embedding_model"], f"{EMBEDDING_MODEL_NAME}@{EMBEDDING_MODEL_REVISION}")
        self.assertRegex(metadata["symptom_model_sha256"], r"^[0-9a-f]{64}$")

    def test_why_plain_has_no_digit_or_percent_and_why_no_longer_says_confidence_boosted(self):
        differential = self._evaluate()["diagnostic_summary"]["differential"]
        self.assertGreaterEqual(len(differential), 3)
        for candidate in differential:
            plain = candidate["why_plain"]
            self.assertTrue(plain)
            self.assertNotRegex(plain, r"[0-9%]")
            self.assertNotIn("confidence", candidate["why"].lower())
            self.assertRegex(candidate["why"], r"score adjusted to \d+\.\d%")

    def test_health_reports_build_commit_and_defaults_to_unknown(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BUILD_COMMIT", None)
            self.assertEqual(self.client.get("/health").json()["build_commit"], "unknown")
        with mock.patch.dict(os.environ, {"BUILD_COMMIT": "abc1234"}):
            health = self.client.get("/health").json()
        self.assertEqual(health["build_commit"], "abc1234")
        self.assertIn("model_version", health)


if __name__ == "__main__":
    unittest.main()

"""
Regression guards for F6C-05 and F6C-01: `model_metadata.calibrated` and every
served `model_version` must be DERIVED from the model_meta.json artifact,
never a literal. Run directly: python tests/test_calibrated_metadata.py
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fastapi.testclient import TestClient

import app as app_module

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMERGENCY_PAYLOAD = {
    "case_token": "case_calibration_test",
    "age": 62,
    "sex": "M",
    "systolic_bp": 195.0,
    "diastolic_bp": 115.0,
    "bmi": 28.4,
    "heart_rate": 112.0,
    "random_glucose": 240.0,
    "spo2": 88.0,
}

ROUTINE_PAYLOAD = {
    "case_token": "case_calibration_test_routine",
    "age": 35,
    "sex": "F",
    "systolic_bp": 120.0,
    "diastolic_bp": 80.0,
    "bmi": 22.0,
    "heart_rate": 75.0,
    "random_glucose": 90.0,
    "spo2": 98.0,
}


class CalibratedFieldMatchesArtifact(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)
        with open(os.path.join(PROJECT_ROOT, "models", "model_meta.json")) as f:
            self.artifact_meta = json.load(f)

    def test_emergency_branch_calibrated_equals_artifact(self):
        artifact_value = self.artifact_meta.get("calibration_used_for_evaluation")
        response = self.client.post("/v1/assess", json=EMERGENCY_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_metadata"]["calibrated"], artifact_value)

    def test_routine_branch_calibrated_equals_artifact(self):
        artifact_value = self.artifact_meta.get("calibration_used_for_evaluation")
        response = self.client.post("/v1/assess", json=ROUTINE_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_metadata"]["calibrated"], artifact_value)

    def test_a_hardcoded_mismatch_would_be_caught(self):
        """The one that matters: proves the assertion above has teeth. Simulates
        the exact F6C-05 bug (module re-hardcodes `calibrated = True`) against the
        real artifact, which records False, and checks the comparison fails."""
        artifact_value = self.artifact_meta.get("calibration_used_for_evaluation")
        original = app_module.calibrated
        try:
            app_module.calibrated = True  # reintroduce the old literal
            self.assertNotEqual(True, artifact_value, "fixture artifact must actually say False for this test to mean anything")
            response = self.client.post("/v1/assess", json=EMERGENCY_PAYLOAD)
            with self.assertRaises(AssertionError):
                self.assertEqual(response.json()["model_metadata"]["calibrated"], artifact_value)
        finally:
            app_module.calibrated = original

    def test_absent_key_reports_none_not_a_default(self):
        """If the artifact ever drops the key, the response must say so honestly
        (null) rather than silently defaulting to True or False."""
        original = app_module.calibrated
        try:
            app_module.calibrated = {}.get("calibration_used_for_evaluation")  # None
            response = self.client.post("/v1/assess", json=EMERGENCY_PAYLOAD)
            self.assertIsNone(response.json()["model_metadata"]["calibrated"])
        finally:
            app_module.calibrated = original


class ModelVersionMatchesArtifact(unittest.TestCase):
    """F6C-01 floor test: the served model_version must always be the one
    the artifact actually names, on both surfaces that report it."""

    def setUp(self):
        self.client = TestClient(app_module.app)
        with open(os.path.join(PROJECT_ROOT, "models", "model_meta.json")) as f:
            self.artifact_meta = json.load(f)

    def test_health_endpoint_model_version_equals_artifact(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_version"], self.artifact_meta["model_version"])

    def test_assess_model_version_equals_artifact(self):
        response = self.client.post("/v1/assess", json=EMERGENCY_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["model_metadata"]["version"],
            self.artifact_meta["model_version"],
        )

    def test_a_hardcoded_version_mismatch_would_be_caught(self):
        """Proves the assertion above has teeth: simulate app.py reverting to
        a literal model_version and check the comparison fails."""
        original = app_module.meta["model_version"]
        try:
            app_module.meta["model_version"] = "xgb-2026-06-11"  # a stale literal
            response = self.client.get("/health")
            with self.assertRaises(AssertionError):
                self.assertEqual(response.json()["model_version"], self.artifact_meta["model_version"])
        finally:
            app_module.meta["model_version"] = original


if __name__ == "__main__":
    unittest.main()

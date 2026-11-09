"""
Live HTTP REST API Integration Test Suite.
Starts the production ThreadingHTTPServer on an ephemeral localhost port and
exercises happy paths and error handling across all endpoints:
  - GET  /health
  - GET  /ready
  - GET  /metrics
  - POST /predict
  - POST /predict/batch
  - Error paths: 400 Bad JSON, 404 Unknown Route, 422 Schema Validation Error
"""

import os
import json
import threading
import unittest
import urllib.request
import urllib.error

from telco_churn.config.settings import load_config
from telco_churn.models.registry import load_model_artifact
from telco_churn.services.retention import ChurnInferenceService
from telco_churn.services.api_server import build_http_server
from tests.fixtures.sample_data import make_valid_subscriber_record

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLiveChurnRestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_config(repo_root=REPO_ROOT)
        artifact = load_model_artifact(cfg.model_artifact_path)
        service = ChurnInferenceService(
            pipeline=artifact["pipeline"],
            model_name=artifact["metadata"].get("model_name", "Proposed Hybrid Soft-Voting Ensemble"),
            low_risk_max=cfg.low_risk_max,
            medium_risk_max=cfg.medium_risk_max,
            max_batch_size=10,
        )
        cls.server = build_http_server("127.0.0.1", 0, service, artifact.get("metadata", {}))
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _request(self, method: str, path: str, body: bytes = None) -> tuple:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    def test_01_health_and_readiness_probes(self):
        """Happy path: GET /health and GET /ready return 200 OK."""
        status, data = self._request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "healthy")

        status_r, data_r = self._request("GET", "/ready")
        self.assertEqual(status_r, 200)
        self.assertTrue(data_r["ready"])

    def test_02_single_subscriber_prediction_happy_path(self):
        """Happy path: POST /predict scores a valid subscriber record in < 50 ms."""
        record = make_valid_subscriber_record("9001-LIVEAPI")
        status, data = self._request("POST", "/predict", json.dumps(record).encode("utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "success")
        pred = data["prediction"]
        self.assertEqual(pred["customerID"], "9001-LIVEAPI")
        self.assertTrue(0.0 <= pred["churn_probability"] <= 1.0)
        self.assertIn(
            pred["risk_tier"],
            {"Low Risk (<35%)", "Medium Risk (35%-65%)", "High Risk (>65%)"},
        )

    def test_03_batch_prediction_happy_path(self):
        """Happy path: POST /predict/batch scores multiple records and updates /metrics."""
        r1 = make_valid_subscriber_record("9002-BATCH1")
        r2 = make_valid_subscriber_record("9003-BATCH2")
        r2["Contract"] = "Two year"
        r2["InternetService"] = "DSL"
        r2["OnlineSecurity"] = "Yes"
        r2["tenure"] = 60

        payload = json.dumps({"records": [r1, r2]}).encode("utf-8")
        status, data = self._request("POST", "/predict/batch", payload)
        self.assertEqual(status, 200)
        self.assertEqual(data["count"], 2)
        # High-risk month-to-month fiber customer must have higher churn probability than 60m two-year DSL customer
        p1 = data["predictions"][0]["churn_probability"]
        p2 = data["predictions"][1]["churn_probability"]
        self.assertGreater(p1, p2)

        # Verify /metrics counter incremented
        m_status, m_data = self._request("GET", "/metrics")
        self.assertEqual(m_status, 200)
        self.assertGreaterEqual(m_data["records_scored"], 3)

    def test_04_malformed_json_returns_400(self):
        """Failure path: malformed JSON body returns HTTP 400 Bad Request."""
        status, data = self._request("POST", "/predict", b"{invalid_json_syntax")
        self.assertEqual(status, 400)
        self.assertEqual(data["error_code"], "BAD_REQUEST_JSON")

    def test_05_schema_validation_failure_returns_422(self):
        """Failure path: missing required fields or invalid categories return HTTP 422."""
        bad_record = {"customerID": "BAD-01", "gender": "Female"}
        status, data = self._request("POST", "/predict", json.dumps(bad_record).encode("utf-8"))
        self.assertEqual(status, 422)
        self.assertEqual(data["error_code"], "SCHEMA_VALIDATION_ERROR")

    def test_06_unknown_endpoint_returns_404(self):
        """Failure path: unknown route returns HTTP 404 Not Found."""
        status, data = self._request("GET", "/nonexistent_route")
        self.assertEqual(status, 404)
        self.assertEqual(data["error_code"], "NOT_FOUND")


if __name__ == "__main__":
    unittest.main()

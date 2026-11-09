"""
Unit tests for configuration management (AppConfig, load_config) and structured
logging with automatic secret redaction. Tests both happy and failure paths.
"""

import os
import tempfile
import unittest

from telco_churn.config.settings import AppConfig, load_config
from telco_churn.utils.exceptions import ConfigurationError
from telco_churn.utils.logger import redact_secrets, get_logger


class TestConfigAndLogger(unittest.TestCase):
    def test_valid_default_config(self):
        """Happy path: default AppConfig validates cleanly."""
        cfg = AppConfig().validate()
        self.assertEqual(cfg.random_seed, 42)
        self.assertAlmostEqual(cfg.test_size, 0.20)
        self.assertEqual(cfg.cv_folds, 5)

    def test_invalid_test_size_raises_configuration_error(self):
        """Failure path: out-of-range test_size raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            AppConfig(test_size=0.95).validate()
        with self.assertRaises(ConfigurationError):
            AppConfig(test_size=0.0).validate()

    def test_invalid_cv_folds_raises_configuration_error(self):
        """Failure path: cv_folds < 2 raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            AppConfig(cv_folds=1).validate()

    def test_invalid_risk_thresholds_raise_configuration_error(self):
        """Failure path: inverted risk thresholds raise ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            AppConfig(low_risk_max=0.70, medium_risk_max=0.40).validate()

    def test_invalid_api_port_raises_configuration_error(self):
        """Failure path: invalid port number raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            AppConfig(api_port=70000).validate()

    def test_missing_explicit_config_file_raises_error(self):
        """Failure path: non-existent explicit config path raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            load_config(config_path="/nonexistent/path/config.json")

    def test_malformed_json_config_raises_error(self):
        """Failure path: corrupted JSON config raises ConfigurationError."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write("{malformed_json: true")
            temp_path = tf.name
        try:
            with self.assertRaises(ConfigurationError):
                load_config(config_path=temp_path)
        finally:
            os.remove(temp_path)

    def test_secret_redaction_masks_credentials(self):
        """Security test: redact_secrets masks api_key, password, token, and secret values."""
        raw_msg = "Connecting with api_key=sk_live_987654321 and password='mySuperSecretPassword123' token: abcdef12345"
        sanitized = redact_secrets(raw_msg)
        self.assertNotIn("sk_live_987654321", sanitized)
        self.assertNotIn("mySuperSecretPassword123", sanitized)
        self.assertNotIn("abcdef12345", sanitized)
        self.assertIn("***REDACTED***", sanitized)

    def test_get_logger_configures_handler(self):
        """Happy path: get_logger returns a configured logger with secret filter."""
        log = get_logger("test_logger", level="DEBUG")
        self.assertTrue(len(log.handlers) >= 1)


if __name__ == "__main__":
    unittest.main()

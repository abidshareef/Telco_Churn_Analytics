"""
Unit tests for data ingestion, SHA-256 integrity hashing, schema validation,
duplicate ID detection, domain range validation, and data leakage guards.
"""

import os
import tempfile
import unittest
import pandas as pd

from telco_churn.data.ingestion import ingest_raw_csv, compute_file_sha256
from telco_churn.data.validation import validate_raw_dataset, assert_no_data_leakage
from telco_churn.utils.exceptions import DataIngestionError, SchemaValidationError
from tests.fixtures.sample_data import make_synthetic_telco_df


class TestDataIngestionAndValidation(unittest.TestCase):
    def setUp(self):
        self.valid_df = make_synthetic_telco_df(n_rows=60, seed=42)

    def test_ingest_missing_file_raises_error(self):
        """Failure path: non-existent CSV path raises DataIngestionError."""
        with self.assertRaises(DataIngestionError):
            ingest_raw_csv("/tmp/does_not_exist_telco_12345.csv")

    def test_ingest_empty_file_raises_error(self):
        """Failure path: 0-byte CSV file raises DataIngestionError."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
            temp_path = tf.name
        try:
            with self.assertRaises(DataIngestionError):
                ingest_raw_csv(temp_path)
        finally:
            os.remove(temp_path)

    def test_ingest_valid_csv_and_checksum(self):
        """Happy path: valid CSV ingests cleanly and returns SHA-256 checksum."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
            self.valid_df.to_csv(tf.name, index=False)
            temp_path = tf.name
        try:
            df_loaded, meta = ingest_raw_csv(temp_path)
            self.assertEqual(len(df_loaded), 60)
            self.assertEqual(meta["n_columns"], 21)
            self.assertEqual(len(meta["sha256_checksum"]), 64)
            self.assertEqual(meta["sha256_checksum"], compute_file_sha256(temp_path))
        finally:
            os.remove(temp_path)

    def test_validate_valid_dataset_passes(self):
        """Happy path: valid synthetic dataset passes schema validation."""
        report = validate_raw_dataset(self.valid_df, require_target=True)
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["whitespace_total_charges_count"], 2)

    def test_validate_missing_columns_raises_error(self):
        """Failure path: dropping required column Contract raises SchemaValidationError."""
        bad_df = self.valid_df.drop(columns=["Contract"])
        with self.assertRaises(SchemaValidationError) as ctx:
            validate_raw_dataset(bad_df)
        self.assertIn("Contract", str(ctx.exception))

    def test_validate_duplicate_customer_ids_raises_error(self):
        """Failure path: duplicate customerID values raise SchemaValidationError."""
        dup_df = self.valid_df.copy()
        dup_df.loc[1, "customerID"] = dup_df.loc[0, "customerID"]
        with self.assertRaises(SchemaValidationError):
            validate_raw_dataset(dup_df)

    def test_validate_invalid_categorical_value_raises_error(self):
        """Failure path: invalid InternetService category raises SchemaValidationError."""
        bad_cat = self.valid_df.copy()
        bad_cat.loc[0, "InternetService"] = "Satellite_5G_Invalid"
        with self.assertRaises(SchemaValidationError):
            validate_raw_dataset(bad_cat)

    def test_validate_negative_monthly_charges_raises_error(self):
        """Failure path: non-positive MonthlyCharges raises SchemaValidationError."""
        bad_num = self.valid_df.copy()
        bad_num.loc[0, "MonthlyCharges"] = -15.0
        with self.assertRaises(SchemaValidationError):
            validate_raw_dataset(bad_num)

    def test_data_leakage_guard_detects_target_and_overlap(self):
        """Failure path: assert_no_data_leakage catches target column or overlapping indices."""
        X_tr = self.valid_df.iloc[:40].drop(columns=["customerID", "Churn"])
        X_te = self.valid_df.iloc[40:].drop(columns=["customerID", "Churn"])
        # Should pass when clean
        assert_no_data_leakage(X_tr, X_te)

        # Overlapping index should fail
        X_te_overlap = self.valid_df.iloc[35:55].drop(columns=["customerID", "Churn"])
        with self.assertRaises(SchemaValidationError):
            assert_no_data_leakage(X_tr, X_te_overlap)

        # Target column left in X_tr should fail
        X_tr_leaked = self.valid_df.iloc[:40]
        with self.assertRaises(SchemaValidationError):
            assert_no_data_leakage(X_tr_leaked, X_te)


if __name__ == "__main__":
    unittest.main()

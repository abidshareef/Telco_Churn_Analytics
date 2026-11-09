"""
Integration test verifying the complete end-to-end training & evaluation pipeline
on the canonical 7,043-record Telco Customer Churn dataset.
"""

import os
import unittest
import pandas as pd

from telco_churn.config.settings import load_config
from telco_churn.data.ingestion import compute_file_sha256
from telco_churn.pipelines.training_pipeline import ChurnTrainingOrchestrator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEndToEndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(repo_root=REPO_ROOT)
        cls.raw_sha_before = compute_file_sha256(cls.config.raw_data_path)
        cls.orchestrator = ChurnTrainingOrchestrator(config=cls.config)
        cls.report = cls.orchestrator.run(generate_plots=True)
        cls.raw_sha_after = compute_file_sha256(cls.config.raw_data_path)

    def test_01_raw_dataset_immutability(self):
        """Verify raw dataset file was never modified during pipeline execution."""
        self.assertEqual(self.raw_sha_before, self.raw_sha_after)
        self.assertTrue(self.report["raw_data_immutability_verified"])

    def test_02_interim_and_processed_datasets_generated(self):
        """Verify interim cleaned CSV and processed train/test/prediction CSVs exist and match row counts."""
        artifacts = self.report["artifacts"]
        for key in ["interim_csv", "train_csv", "test_csv", "predictions_csv", "model_artifact"]:
            self.assertTrue(os.path.exists(artifacts[key]), f"Missing artifact: {artifacts[key]}")

        interim_df = pd.read_csv(artifacts["interim_csv"])
        train_df = pd.read_csv(artifacts["train_csv"])
        test_df = pd.read_csv(artifacts["test_csv"])
        pred_df = pd.read_csv(artifacts["predictions_csv"])

        self.assertEqual(len(interim_df), 7043)
        self.assertEqual(len(train_df) + len(test_df), 7043)
        self.assertEqual(len(pred_df), len(test_df))

    def test_03_benchmark_performance_and_best_model(self):
        """Verify Proposed Hybrid Soft-Voting Ensemble achieves ROC-AUC >= 0.94 and F1 >= 0.78."""
        best_row = self.report["model_benchmarks"][0]
        self.assertIn("Proposed Hybrid Soft-Voting Ensemble", best_row["Model"])
        self.assertGreaterEqual(best_row["Test_ROC_AUC"], 0.94)
        self.assertGreaterEqual(best_row["Test_PR_AUC"], 0.86)
        self.assertGreaterEqual(best_row["Test_F1_Score"], 0.78)
        self.assertGreaterEqual(best_row["Test_Accuracy"], 0.86)

    def test_04_roi_and_diagnostic_figures(self):
        """Verify positive net ROI and non-empty 300-DPI diagnostic figures."""
        roi = self.report["retention_roi_simulation"]
        self.assertGreater(roi["Net_Annual_Benefit_USD"], 50000.0)
        self.assertGreater(roi["Target_Capture_Recall_Rate"], 0.85)
        for fig_path in self.report["artifacts"]["figures"]:
            self.assertTrue(os.path.exists(fig_path))
            self.assertGreater(os.path.getsize(fig_path), 50000)


if __name__ == "__main__":
    unittest.main()

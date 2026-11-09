"""
Unit tests for ML classifiers, StratifiedKFold, model persistence registry,
and 3-Tier Retention Risk Segmentation & Campaign ROI calculations.
"""

import os
import tempfile
import unittest
import numpy as np

from telco_churn.preprocessing.cleaner import clean_telco_data
from telco_churn.preprocessing.features import engineer_domain_features
from telco_churn.preprocessing.transformer import TabularPreprocessor
from telco_churn.models.classifiers import (
    LogisticRegressionClassifier,
    DecisionTreeCART,
    RandomForestEnsemble,
    GradientBoostingClassifierGBM,
    MLPNeuralNetClassifier,
    WeightedSoftVotingEnsemble,
    SoftVotingPipeline,
)
from telco_churn.models.evaluation import StratifiedKFold, roc_auc_score
from telco_churn.models.registry import save_model_artifact, load_model_artifact
from telco_churn.services.retention import (
    assign_risk_tier_and_action,
    evaluate_retention_roi,
    ChurnInferenceService,
)
from telco_churn.utils.exceptions import (
    ModelNotFittedError,
    ModelPersistenceError,
    InferenceError,
)
from tests.fixtures.sample_data import (
    make_synthetic_telco_df,
    make_valid_subscriber_record,
)


class TestModelsAndRetentionROI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_df = make_synthetic_telco_df(n_rows=120, seed=42)
        cleaned, _ = clean_telco_data(raw_df)
        fe = engineer_domain_features(cleaned)
        cls.X = fe.drop(columns=["customerID", "Churn", "Churn_Binary"])
        cls.y = fe["Churn_Binary"].values
        cls.pre = TabularPreprocessor()
        cls.X_mat = cls.pre.fit_transform(cls.X)

    def test_unfitted_classifiers_raise_model_not_fitted_error(self):
        """Failure path: calling predict_proba before fit raises ModelNotFittedError."""
        for clf in [
            LogisticRegressionClassifier(),
            DecisionTreeCART(),
            RandomForestEnsemble(n_estimators=5),
            GradientBoostingClassifierGBM(n_estimators=5),
            MLPNeuralNetClassifier(epochs=2),
        ]:
            with self.assertRaises(ModelNotFittedError):
                clf.predict_proba(self.X_mat)

    def test_all_classifiers_fit_and_output_valid_probabilities(self):
        """Happy path: all classifiers fit and return calibrated probabilities in [0, 1]."""
        lr = LogisticRegressionClassifier(C=1.0).fit(self.X_mat, self.y)
        dt = DecisionTreeCART(max_depth=4).fit(self.X_mat, self.y)
        rf = RandomForestEnsemble(n_estimators=10, max_depth=5).fit(self.X_mat, self.y)
        gbm = GradientBoostingClassifierGBM(n_estimators=15, max_depth=3).fit(self.X_mat, self.y)
        mlp = MLPNeuralNetClassifier(epochs=10).fit(self.X_mat, self.y)

        for clf in [lr, dt, rf, gbm, mlp]:
            probs = clf.predict_proba(self.X_mat)
            self.assertEqual(probs.shape, (len(self.y), 2))
            self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
            self.assertGreater(roc_auc_score(self.y, probs[:, 1]), 0.75)

    def test_stratified_kfold_preserves_class_balance(self):
        """Happy path: StratifiedKFold preserves class proportions across all 5 splits."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        overall_rate = self.y.mean()
        for tr_idx, val_idx in skf.split(self.X, self.y):
            self.assertEqual(len(set(tr_idx).intersection(set(val_idx))), 0)
            self.assertAlmostEqual(self.y[val_idx].mean(), overall_rate, delta=0.05)

    def test_model_registry_save_load_and_corrupt_handling(self):
        """Tests saving and loading a fitted pipeline and handling missing/corrupt files."""
        pipe = SoftVotingPipeline(
            TabularPreprocessor(), LogisticRegressionClassifier(C=1.0), optimal_threshold=0.48
        ).fit(self.X, self.y)

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path = os.path.join(tmpdir, "model.pkl")
            save_model_artifact(pipe, art_path, metadata={"test_auc": 0.91})
            loaded = load_model_artifact(art_path)
            self.assertEqual(loaded["metadata"]["test_auc"], 0.91)
            probs = loaded["pipeline"].predict_proba(self.X)[:, 1]
            self.assertEqual(len(probs), len(self.y))

            # Corrupt file should raise ModelPersistenceError
            corrupt_path = os.path.join(tmpdir, "corrupt.pkl")
            with open(corrupt_path, "wb") as f:
                f.write(b"not_a_valid_pickle_stream")
            with self.assertRaises(ModelPersistenceError):
                load_model_artifact(corrupt_path)

    def test_retention_risk_tiering_and_inference_service(self):
        """Tests risk tier assignment, ROI computation, and batch limit enforcement."""
        tier_low, _ = assign_risk_tier_and_action(0.15)
        tier_med, _ = assign_risk_tier_and_action(0.50)
        tier_high, _ = assign_risk_tier_and_action(0.82)
        self.assertEqual(tier_low, "Low Risk (<35%)")
        self.assertEqual(tier_med, "Medium Risk (35%-65%)")
        self.assertEqual(tier_high, "High Risk (>65%)")

        pipe = SoftVotingPipeline(
            TabularPreprocessor(), LogisticRegressionClassifier(C=1.0)
        ).fit(self.X, self.y)
        svc = ChurnInferenceService(pipe, max_batch_size=3)

        rec = make_valid_subscriber_record()
        res = svc.predict_records([rec])
        self.assertEqual(res["count"], 1)
        self.assertIn("churn_probability", res["predictions"][0])

        # Exceeding max_batch_size should raise InferenceError
        with self.assertRaises(InferenceError):
            svc.predict_records([rec, rec, rec, rec])


if __name__ == "__main__":
    unittest.main()

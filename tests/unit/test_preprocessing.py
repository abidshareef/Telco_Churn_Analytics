"""
Unit tests for data cleaning, domain feature engineering, zero-leakage
TabularPreprocessor, and statistical hypothesis tests.
"""

import unittest
import numpy as np
import pandas as pd

from telco_churn.preprocessing.cleaner import clean_telco_data
from telco_churn.preprocessing.features import (
    engineer_domain_features,
    ENGINEERED_FEATURE_NAMES,
)
from telco_churn.preprocessing.transformer import (
    TabularPreprocessor,
    run_statistical_hypothesis_tests,
)
from telco_churn.utils.exceptions import PreprocessingError, ModelNotFittedError
from tests.fixtures.sample_data import make_synthetic_telco_df


class TestPreprocessingAndFeatures(unittest.TestCase):
    def setUp(self):
        self.raw_df = make_synthetic_telco_df(n_rows=80, seed=42)

    def test_clean_telco_data_handles_whitespace_and_preserves_input(self):
        """Happy path: clean_telco_data fixes blank TotalCharges without mutating input."""
        raw_copy = self.raw_df.copy(deep=True)
        cleaned, summary = clean_telco_data(self.raw_df, include_target=True)

        # Input DataFrame must remain untouched
        self.assertTrue(self.raw_df.equals(raw_copy))
        self.assertEqual(summary["n_whitespace_total_charges_fixed"], 2)
        self.assertEqual(cleaned["TotalCharges"].isna().sum(), 0)
        self.assertIn("Churn_Binary", cleaned.columns)
        self.assertEqual(set(cleaned["Churn_Binary"].unique()), {0, 1})

    def test_clean_empty_dataframe_raises_error(self):
        """Failure path: empty DataFrame raises PreprocessingError."""
        with self.assertRaises(PreprocessingError):
            clean_telco_data(pd.DataFrame())

    def test_engineer_domain_features_creates_all_six_attributes(self):
        """Happy path: engineer_domain_features creates all 6 features within valid bounds."""
        cleaned, _ = clean_telco_data(self.raw_df)
        fe = engineer_domain_features(cleaned)
        for col in ENGINEERED_FEATURE_NAMES:
            self.assertIn(col, fe.columns)
            self.assertEqual(fe[col].isna().sum(), 0)

        self.assertTrue(fe["Total_Services_Count"].between(0, 8).all())
        self.assertTrue(fe["Security_Support_Count"].between(0, 4).all())
        self.assertTrue(fe["Charge_Spike_Ratio"].between(0.5, 2.5).all())
        self.assertTrue(fe["Contract_Risk_Index"].between(0.0, 5.5).all())

    def test_preprocessor_unfitted_transform_raises_error(self):
        """Failure path: calling transform() before fit() raises ModelNotFittedError."""
        cleaned, _ = clean_telco_data(self.raw_df)
        pre = TabularPreprocessor()
        with self.assertRaises(ModelNotFittedError):
            pre.transform(cleaned)

    def test_preprocessor_fit_transform_and_unseen_categories(self):
        """Happy path: TabularPreprocessor standardizes numeric cols and handles unseen categories."""
        cleaned, _ = clean_telco_data(self.raw_df)
        fe = engineer_domain_features(cleaned)
        X = fe.drop(columns=["customerID", "Churn", "Churn_Binary"])

        pre = TabularPreprocessor()
        mat = pre.fit_transform(X)
        self.assertEqual(mat.shape[0], len(X))
        self.assertEqual(mat.shape[1], len(pre.feature_names_out_))
        self.assertFalse(np.isnan(mat).any())

    def test_statistical_hypothesis_tests_execution(self):
        """Happy path: run_statistical_hypothesis_tests computes Mann-Whitney and Chi-Square stats."""
        cleaned, _ = clean_telco_data(self.raw_df)
        stats_df = run_statistical_hypothesis_tests(cleaned)
        self.assertEqual(len(stats_df), 13)
        self.assertIn("Effect_Size", stats_df.columns)


if __name__ == "__main__":
    unittest.main()

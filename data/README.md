# Dataset Documentation & Schema Contract

## 1. Source & Provenance
* **Canonical Source:** [IBM / Kaggle Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Scale:** $7,043$ subscriber records $\times$ $21$ attributes ($19$ predictors, $1$ primary identifier `customerID`, $1$ target `Churn`).
* **Class Distribution:** $5,174$ Retained (`No`, $73.46\%$) vs. $1,869$ Churned (`Yes`, $26.54\%$).

## 2. Directory Conventions & Immutability Policy
```text
data/
├── raw/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IMMUTABLE raw dataset (verified via SHA-256)
├── interim/
│   └── cleaned_telco_churn.csv                # Coerced TotalCharges & zero-tenure imputed
└── processed/
    ├── train_engineered.csv                   # 80% Stratified Training Split (n = 5,634)
    ├── test_engineered.csv                    # 20% Stratified Hold-Out Split (n = 1,409)
    └── customer_risk_predictions.csv          # Hold-out predictions, risk tiers & CRM actions
```
* **Strict Immutability:** `src/telco_churn/data/ingestion.py` records the SHA-256 digest of `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` prior to ingestion and `src/telco_churn/pipelines/training_pipeline.py` verifies that the raw file's SHA-256 digest is identical after pipeline completion.

## 3. Edge Cases & Cleaning Contract
* **Whitespace Strings in `TotalCharges`:** Exactly $11$ records correspond to newly onboarded customers (`tenure == 0`) where `TotalCharges` is stored as `" "` (a single whitespace character). Instead of dropping these valid customers ([Wei, 2026](https://www.researchgate.net/publication/398104166_Machine_Learning-Based_Prediction_of_Telecom_Customer_Churn_Comparative_Model_Analysis)), `clean_telco_data()` coerces `" "` to `NaN` and deterministically imputes `0.0` for `tenure == 0` (and `tenure * MonthlyCharges` as a general fallback).
* **Zero Leakage Guarantee:** `assert_no_data_leakage()` ensures `customerID`, `Churn`, and `Churn_Binary` are removed prior to feature matrix construction and verifies zero index overlap between `X_train` and `X_test`.

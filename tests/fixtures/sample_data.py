"""
Deterministic synthetic test fixtures for unit and integration tests.
Provides both valid subscriber records (including zero-tenure whitespace edge cases)
and intentionally malformed payloads for failure-path testing.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


def make_valid_subscriber_record(customer_id: str = "1001-TESTA") -> Dict[str, Any]:
    """Returns a single valid subscriber dictionary matching the schema contract."""
    return {
        "customerID": customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 6,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": "537.00",
        "Churn": "Yes",
    }


def make_synthetic_telco_df(n_rows: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generates a small, deterministic, schema-valid DataFrame with 2 blank TotalCharges rows."""
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []
    for i in range(n_rows):
        is_churn = (i % 3 == 0)
        tenure_val = 0 if i in (0, 1) else int(rng.integers(1, 70))
        mc_val = round(float(rng.uniform(25.0, 105.0)), 2)
        tc_str = " " if tenure_val == 0 else f"{tenure_val * mc_val:.2f}"
        records.append(
            {
                "customerID": f"{1000 + i}-SYNTH",
                "gender": "Female" if i % 2 == 0 else "Male",
                "SeniorCitizen": 1 if i % 5 == 0 else 0,
                "Partner": "Yes" if i % 2 == 0 else "No",
                "Dependents": "No" if is_churn else "Yes",
                "tenure": tenure_val,
                "PhoneService": "Yes",
                "MultipleLines": "Yes" if i % 3 == 0 else "No",
                "InternetService": "Fiber optic" if is_churn else "DSL",
                "OnlineSecurity": "No" if is_churn else "Yes",
                "OnlineBackup": "Yes" if i % 2 == 0 else "No",
                "DeviceProtection": "No",
                "TechSupport": "No" if is_churn else "Yes",
                "StreamingTV": "Yes" if i % 2 == 0 else "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month" if is_churn else "Two year",
                "PaperlessBilling": "Yes" if is_churn else "No",
                "PaymentMethod": "Electronic check" if is_churn else "Bank transfer (automatic)",
                "MonthlyCharges": mc_val,
                "TotalCharges": tc_str,
                "Churn": "Yes" if (is_churn and tenure_val > 0) else "No",
            }
        )
    return pd.DataFrame(records)

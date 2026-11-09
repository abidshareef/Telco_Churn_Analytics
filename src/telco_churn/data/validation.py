"""
Dataset schema contract validation, duplicate detection, domain range checks,
and target leakage guards.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from telco_churn.utils.exceptions import SchemaValidationError
from telco_churn.utils.logger import get_logger

logger = get_logger("telco_churn.data.validation")

REQUIRED_RAW_COLUMNS: List[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

REQUIRED_INFERENCE_COLUMNS: List[str] = [
    c for c in REQUIRED_RAW_COLUMNS if c not in ("customerID", "Churn")
]

ALLOWED_CATEGORICAL_DOMAINS: Dict[str, set] = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1, "0", "1"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


def validate_raw_dataset(df: pd.DataFrame, require_target: bool = True) -> Dict[str, Any]:
    """
    Validates a DataFrame against the Telco Customer Churn schema contract.
    Raises SchemaValidationError if critical contract violations are found.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaValidationError("Input must be a pandas DataFrame.")
    if df.empty:
        raise SchemaValidationError("Dataset cannot be empty (0 rows).")

    expected_cols = REQUIRED_RAW_COLUMNS if require_target else REQUIRED_INFERENCE_COLUMNS
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise SchemaValidationError(
            f"Dataset is missing required columns: {missing_cols}",
            context={"missing_columns": missing_cols},
        )

    # Check for duplicate customerIDs when customerID column is present
    duplicate_ids = 0
    if "customerID" in df.columns:
        if df["customerID"].isna().any():
            raise SchemaValidationError("customerID column contains null/NaN values.")
        duplicate_ids = int(df["customerID"].duplicated().sum())
        if duplicate_ids > 0:
            raise SchemaValidationError(
                f"Detected {duplicate_ids} duplicate customerID records.",
                context={"duplicate_customer_ids": duplicate_ids},
            )

    # Validate numeric ranges for tenure and MonthlyCharges
    try:
        tenure_num = pd.to_numeric(df["tenure"], errors="raise")
        monthly_num = pd.to_numeric(df["MonthlyCharges"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise SchemaValidationError(
            f"Non-numeric values found in tenure or MonthlyCharges: {exc}"
        ) from exc

    if tenure_num.isna().any() or (tenure_num < 0).any() or (tenure_num > 150).any():
        raise SchemaValidationError(
            "Invalid tenure values detected: tenure must be between 0 and 150 months."
        )
    if monthly_num.isna().any() or (monthly_num <= 0).any() or (monthly_num > 500).any():
        raise SchemaValidationError(
            "Invalid MonthlyCharges values detected: must be in (0, 500]."
        )

    # Validate categorical domains
    for col, allowed_set in ALLOWED_CATEGORICAL_DOMAINS.items():
        if col not in df.columns:
            continue
        actual_vals = set(df[col].dropna().unique())
        invalid_vals = actual_vals - allowed_set
        if invalid_vals:
            raise SchemaValidationError(
                f"Column '{col}' contains unexpected categorical values: {invalid_vals}",
                context={"column": col, "invalid_values": list(map(str, invalid_vals))},
            )

    # Audit TotalCharges whitespace / non-numeric entries
    coerced_tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank_tc = int(coerced_tc.isna().sum())
    if (coerced_tc.dropna() < 0).any():
        raise SchemaValidationError("Negative values found in TotalCharges.")

    # Outlier audit via IQR on MonthlyCharges & tenure
    q1_mc, q3_mc = np.percentile(monthly_num, [25, 75])
    iqr_mc = q3_mc - q1_mc
    n_outliers_mc = int(
        ((monthly_num < (q1_mc - 1.5 * iqr_mc)) | (monthly_num > (q3_mc + 1.5 * iqr_mc))).sum()
    )

    report = {
        "is_valid": True,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicate_customer_ids": duplicate_ids,
        "whitespace_total_charges_count": n_blank_tc,
        "monthly_charges_iqr_outliers": n_outliers_mc,
    }
    logger.info(
        f"Schema validation passed ({len(df)} rows, {n_blank_tc} blank TotalCharges flagged for cleaning)."
    )
    return report


def assert_no_data_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    forbidden_cols: List[str] = None,
) -> None:
    """
    Verifies zero index overlap between train and test splits and ensures target
    or identifier columns are excluded from feature matrices.
    """
    forbidden = forbidden_cols or ["customerID", "Churn", "Churn_Binary"]
    for col in forbidden:
        if col in X_train.columns or col in X_test.columns:
            raise SchemaValidationError(
                f"Target or identifier leakage detected: column '{col}' present in feature matrix.",
                context={"leaked_column": col},
            )

    overlap = set(X_train.index).intersection(set(X_test.index))
    if overlap:
        raise SchemaValidationError(
            f"Train/test index leakage detected: {len(overlap)} overlapping indices.",
            context={"overlap_count": len(overlap)},
        )

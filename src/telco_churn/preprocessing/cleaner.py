"""
Immutable dataset cleaning module.
Coerces blank whitespace strings in TotalCharges, handles zero-tenure onboarding
records, and encodes binary target labels without modifying the input DataFrame in place.
"""

from typing import Tuple, Dict, Any
import pandas as pd

from telco_churn.utils.exceptions import PreprocessingError
from telco_churn.utils.logger import get_logger

logger = get_logger("telco_churn.preprocessing.cleaner")


def clean_telco_data(df: pd.DataFrame, include_target: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans a validated Telco Customer Churn DataFrame and returns a new cleaned DataFrame
    along with a cleaning summary report.
    """
    if df is None or df.empty:
        raise PreprocessingError("Cannot clean an empty or None DataFrame.")

    cleaned = df.copy(deep=True)

    # Coerce whitespace strings (" ") in TotalCharges to NaN
    raw_tc = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    n_coerced = int(raw_tc.isna().sum())
    cleaned["TotalCharges"] = raw_tc

    # Impute 0.0 for newly onboarded customers (tenure == 0)
    zero_tenure_mask = cleaned["TotalCharges"].isna() & (cleaned["tenure"] == 0)
    n_zero_tenure_fixed = int(zero_tenure_mask.sum())
    cleaned.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    # Fallback imputation for any remaining missing TotalCharges: tenure * MonthlyCharges
    rem_mask = cleaned["TotalCharges"].isna()
    n_fallback_fixed = int(rem_mask.sum())
    if rem_mask.any():
        cleaned.loc[rem_mask, "TotalCharges"] = (
            cleaned.loc[rem_mask, "tenure"] * cleaned.loc[rem_mask, "MonthlyCharges"]
        )

    cleaned["SeniorCitizen"] = pd.to_numeric(cleaned["SeniorCitizen"], errors="raise").astype(int)
    cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="raise").astype(int)
    cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="raise").astype(float)
    cleaned["TotalCharges"] = cleaned["TotalCharges"].astype(float)

    summary: Dict[str, Any] = {
        "n_rows": int(len(cleaned)),
        "n_whitespace_total_charges_fixed": n_coerced,
        "n_zero_tenure_imputed_zero": n_zero_tenure_fixed,
        "n_fallback_imputed": n_fallback_fixed,
    }

    if include_target and "Churn" in cleaned.columns:
        cleaned["Churn_Binary"] = (
            cleaned["Churn"].astype(str).str.strip().str.lower() == "yes"
        ).astype(int)
        summary["overall_churn_rate"] = float(cleaned["Churn_Binary"].mean())
        summary["churn_counts"] = {
            "Retained (0)": int((cleaned["Churn_Binary"] == 0).sum()),
            "Churned (1)": int((cleaned["Churn_Binary"] == 1).sum()),
        }

    logger.info(
        f"Data cleaning complete: fixed {n_coerced} blank TotalCharges entries "
        f"({n_zero_tenure_fixed} zero-tenure records)."
    )
    return cleaned, summary

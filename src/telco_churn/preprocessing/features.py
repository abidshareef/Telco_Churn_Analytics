"""
Domain feature engineering module for Telecom Subscriber Churn Analytics.
Constructs 6 interpretable behavioral and financial interaction features.
"""

from typing import List
import numpy as np
import pandas as pd

from telco_churn.utils.exceptions import PreprocessingError

ADDON_SERVICE_COLS: List[str] = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

PROTECTION_SUPPORT_COLS: List[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
]

ENGINEERED_FEATURE_NAMES: List[str] = [
    "Tenure_Cohort",
    "Total_Services_Count",
    "Security_Support_Count",
    "Charge_Spike_Ratio",
    "Is_Auto_Payment",
    "Contract_Risk_Index",
]


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 6 domain-engineered features from cleaned subscriber data:
      1. Tenure_Cohort: Lifecycle stage ('0-12m_Onboarding', '13-24m_Developing', '25-48m_Mature', '49-72m_Loyal')
      2. Total_Services_Count: Active subscriptions count out of 8 services
      3. Security_Support_Count: Active security/tech-support subscriptions out of 4
      4. Charge_Spike_Ratio: Current MonthlyCharges relative to historical average monthly spend
      5. Is_Auto_Payment: 1 if automatic payment method, else 0
      6. Contract_Risk_Index: Weighted vulnerability score capturing Month-to-month + Fiber + Unprotected + E-check
    """
    if df is None or df.empty:
        raise PreprocessingError("Cannot engineer features on an empty DataFrame.")

    required_for_fe = set(ADDON_SERVICE_COLS + ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"])
    missing = required_for_fe - set(df.columns)
    if missing:
        raise PreprocessingError(
            f"Missing columns required for feature engineering: {missing}",
            context={"missing_columns": list(missing)},
        )

    fe = df.copy(deep=True)

    # 1. Lifecycle Tenure Cohort
    fe["Tenure_Cohort"] = pd.cut(
        fe["tenure"],
        bins=[-1, 12, 24, 48, 200],
        labels=["0-12m_Onboarding", "13-24m_Developing", "25-48m_Mature", "49-72m_Loyal"],
    ).astype(str)

    # 2. Total Services Count (0 to 8)
    fe["Total_Services_Count"] = sum(
        (fe[col] == "Yes").astype(int) for col in ADDON_SERVICE_COLS
    )

    # 3. Security & Support Bundle Count (0 to 4)
    fe["Security_Support_Count"] = sum(
        (fe[col] == "Yes").astype(int) for col in PROTECTION_SUPPORT_COLS
    )

    # 4. Charge Spike Ratio (bounded in [0.5, 2.5])
    expected_avg_monthly = np.where(
        fe["tenure"] > 0,
        fe["TotalCharges"] / np.maximum(fe["tenure"], 1),
        fe["MonthlyCharges"],
    )
    fe["Charge_Spike_Ratio"] = np.clip(
        fe["MonthlyCharges"] / np.maximum(expected_avg_monthly, 1.0),
        0.5,
        2.5,
    )

    # 5. Automatic Payment Indicator
    fe["Is_Auto_Payment"] = (
        fe["PaymentMethod"].astype(str).str.contains("automatic", case=False, na=False)
    ).astype(int)

    # 6. Composite Contract & Service Vulnerability Index
    contract_weight = np.where(
        fe["Contract"] == "Month-to-month",
        2.0,
        np.where(fe["Contract"] == "One year", 0.5, 0.0),
    )
    fiber_weight = np.where(fe["InternetService"] == "Fiber optic", 1.5, 0.0)
    unprotected_weight = np.where(
        (fe["InternetService"] != "No") & (fe["Security_Support_Count"] == 0),
        1.25,
        0.0,
    )
    echeck_weight = np.where(fe["PaymentMethod"] == "Electronic check", 0.75, 0.0)
    fe["Contract_Risk_Index"] = (
        contract_weight + fiber_weight + unprotected_weight + echeck_weight
    )

    return fe

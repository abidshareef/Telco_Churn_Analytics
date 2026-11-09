"""
Customer Retention Risk Segmentation, Campaign ROI Simulation, and Real-Time
Inference Service.
"""

import time
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from telco_churn.models.classifiers import SoftVotingPipeline
from telco_churn.data.validation import validate_raw_dataset
from telco_churn.preprocessing.cleaner import clean_telco_data
from telco_churn.preprocessing.features import engineer_domain_features
from telco_churn.utils.exceptions import InferenceError, SchemaValidationError


def assign_risk_tier_and_action(
    prob: float, low_max: float = 0.35, med_max: float = 0.65
) -> Tuple[str, str]:
    """Maps a calibrated churn probability [0, 1] to a 3-Tier Risk label and CRM action."""
    if prob >= med_max:
        return (
            "High Risk (>65%)",
            "1-Year Contract Migration + 15% Bill Credit + Free Online Security",
        )
    if prob >= low_max:
        return (
            "Medium Risk (35%-65%)",
            "Proactive Tech Support Check-in + Auto-Pay Enrollment Incentive",
        )
    return ("Low Risk (<35%)", "Standard Digital Loyalty Rewards")


def evaluate_retention_roi(
    customer_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    monthly_charges: np.ndarray,
    optimal_threshold: float = 0.50,
    low_max: float = 0.35,
    med_max: float = 0.65,
    intervention_cost_usd: float = 45.0,
    retention_save_rate: float = 0.40,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Segments held-out subscribers into Low/Medium/High churn risk tiers and
    computes expected financial campaign ROI.
    """
    y_pred = (y_prob >= optimal_threshold).astype(int)

    tiers = np.where(
        y_prob >= med_max,
        "High Risk (>65%)",
        np.where(y_prob >= low_max, "Medium Risk (35%-65%)", "Low Risk (<35%)"),
    )

    actions = np.where(
        y_prob >= med_max,
        "1-Year Contract Migration + 15% Bill Credit + Free Online Security",
        np.where(
            y_prob >= low_max,
            "Proactive Tech Support Check-in + Auto-Pay Enrollment Incentive",
            "Standard Digital Loyalty Rewards",
        ),
    )

    pred_df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "Actual_Churn": y_true,
            "Predicted_Churn_Prob": np.round(y_prob, 4),
            "Predicted_Churn_Label": y_pred,
            "Risk_Tier": tiers,
            "MonthlyCharges": monthly_charges,
            "Annualized_Revenue_At_Risk": np.round(monthly_charges * 12.0 * y_prob, 2),
            "Recommended_Intervention": actions,
        }
    )

    high_med_mask = y_prob >= low_max
    true_churners_captured = int(((y_true == 1) & high_med_mask).sum())
    total_targeted = int(high_med_mask.sum())

    churner_charges = monthly_charges[y_true == 1]
    avg_annual_rev_churner = float(
        (churner_charges.mean() if len(churner_charges) > 0 else monthly_charges.mean()) * 12.0
    )
    campaign_cost = total_targeted * intervention_cost_usd
    retained_customers = retention_save_rate * true_churners_captured
    gross_revenue_saved = retained_customers * avg_annual_rev_churner
    net_roi_dollars = gross_revenue_saved - campaign_cost
    roi_percentage = (net_roi_dollars / max(campaign_cost, 1.0)) * 100.0

    tier_summary = (
        pred_df.groupby("Risk_Tier")
        .agg(
            Subscriber_Count=("customerID", "count"),
            Actual_Churn_Rate=("Actual_Churn", "mean"),
            Avg_Churn_Probability=("Predicted_Churn_Prob", "mean"),
            Avg_Monthly_Charge=("MonthlyCharges", "mean"),
            Total_Annual_Rev_At_Risk=("Annualized_Revenue_At_Risk", "sum"),
        )
        .reset_index()
    )

    roi_summary = {
        "Test_Cohort_Size": int(len(pred_df)),
        "Total_Actual_Churners_in_Test": int(y_true.sum()),
        "Targeted_Subscribers_Med_High_Risk": total_targeted,
        "True_Churners_Captured_in_Target": true_churners_captured,
        "Target_Capture_Recall_Rate": float(
            true_churners_captured / max(int(y_true.sum()), 1)
        ),
        "Avg_Annual_Revenue_Per_Churner_USD": round(avg_annual_rev_churner, 2),
        "Estimated_Campaign_Cost_USD": round(campaign_cost, 2),
        "Gross_Annual_Revenue_Saved_USD": round(gross_revenue_saved, 2),
        "Net_Annual_Benefit_USD": round(net_roi_dollars, 2),
        "Campaign_ROI_Percent": round(roi_percentage, 1),
        "Tier_Breakdown": tier_summary.to_dict(orient="records"),
    }

    return pred_df, roi_summary


class ChurnInferenceService:
    """
    Stateful inference service wrapping a fitted SoftVotingPipeline to score single
    or batch subscriber records with full input validation and latency tracking.
    """

    def __init__(
        self,
        pipeline: SoftVotingPipeline,
        model_name: str = "Proposed Hybrid Soft-Voting Ensemble",
        low_risk_max: float = 0.35,
        medium_risk_max: float = 0.65,
        max_batch_size: int = 1000,
    ):
        self.pipeline = pipeline
        self.model_name = model_name
        self.low_risk_max = low_risk_max
        self.medium_risk_max = medium_risk_max
        self.max_batch_size = max_batch_size
        self.requests_served: int = 0
        self.records_scored: int = 0

    def predict_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validates, cleans, engineers features, and scores a list of subscriber dicts."""
        t0 = time.perf_counter()
        if not isinstance(records, list) or len(records) == 0:
            raise InferenceError(
                "Input payload must be a non-empty list of subscriber record objects."
            )
        if len(records) > self.max_batch_size:
            raise InferenceError(
                f"Batch size {len(records)} exceeds maximum allowed ({self.max_batch_size}).",
                context={"batch_size": len(records), "max_batch_size": self.max_batch_size},
            )

        raw_df = pd.DataFrame(records)
        if "customerID" not in raw_df.columns:
            raw_df["customerID"] = [f"ANON-{i+1:04d}" for i in range(len(raw_df))]

        # Validate against inference schema contract (target Churn not required)
        validate_raw_dataset(raw_df, require_target=False)

        cleaned_df, _ = clean_telco_data(raw_df, include_target=False)
        fe_df = engineer_domain_features(cleaned_df)

        drop_cols = ["customerID", "Churn", "Churn_Binary"]
        X_infer = fe_df.drop(columns=[c for c in drop_cols if c in fe_df.columns])

        probs = self.pipeline.predict_proba(X_infer)[:, 1]
        thresh = self.pipeline.optimal_threshold

        predictions = []
        for idx, prob in enumerate(probs):
            p_val = float(prob)
            tier, action = assign_risk_tier_and_action(
                p_val, self.low_risk_max, self.medium_risk_max
            )
            mc = float(cleaned_df.iloc[idx]["MonthlyCharges"])
            predictions.append(
                {
                    "customerID": str(cleaned_df.iloc[idx]["customerID"]),
                    "churn_probability": round(p_val, 4),
                    "churn_prediction": int(p_val >= thresh),
                    "decision_threshold": round(float(thresh), 4),
                    "risk_tier": tier,
                    "contract_risk_index": round(float(fe_df.iloc[idx]["Contract_Risk_Index"]), 2),
                    "annual_revenue_at_risk_usd": round(mc * 12.0 * p_val, 2),
                    "recommended_intervention": action,
                }
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.requests_served += 1
        self.records_scored += len(predictions)

        return {
            "model_name": self.model_name,
            "count": len(predictions),
            "latency_ms": round(elapsed_ms, 3),
            "predictions": predictions,
        }

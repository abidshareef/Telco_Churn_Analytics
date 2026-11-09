"""
Zero-leakage tabular preprocessing transformer (median/mode imputation, StandardScaler,
OneHotEncoder) and non-parametric statistical hypothesis testing suite.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats

from telco_churn.utils.exceptions import ModelNotFittedError, PreprocessingError


class TabularPreprocessor:
    """
    Fits imputation statistics, standardization parameters, and one-hot category
    vocabularies strictly on the training fold and transforms unseen data safely.
    """

    def __init__(self):
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.num_medians: Dict[str, float] = {}
        self.num_means: Dict[str, float] = {}
        self.num_stds: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.cat_categories: Dict[str, List[str]] = {}
        self.feature_names_out_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "TabularPreprocessor":
        if X is None or X.empty:
            raise PreprocessingError("Cannot fit TabularPreprocessor on empty DataFrame.")

        self.num_cols = X.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()
        self.cat_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        self.feature_names_out_ = list(self.num_cols)
        for col in self.num_cols:
            med = float(X[col].median())
            filled = X[col].fillna(med).values.astype(float)
            self.num_medians[col] = med
            self.num_means[col] = float(np.mean(filled))
            std_val = float(np.std(filled))
            self.num_stds[col] = std_val if std_val > 1e-9 else 1.0

        for col in self.cat_cols:
            s = X[col].astype(str)
            mode_val = s.mode().iloc[0] if not s.mode().empty else "Missing"
            self.cat_modes[col] = mode_val
            cats = sorted(s.fillna(mode_val).unique().tolist())
            kept_cats = cats[1:] if len(cats) > 1 else cats
            self.cat_categories[col] = kept_cats
            for c_val in kept_cats:
                self.feature_names_out_.append(f"{col}_{c_val}")

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise ModelNotFittedError(
                "TabularPreprocessor must be fitted via fit() before calling transform()."
            )
        if X is None or X.empty:
            raise PreprocessingError("Cannot transform an empty DataFrame.")

        missing = set(self.num_cols + self.cat_cols) - set(X.columns)
        if missing:
            raise PreprocessingError(
                f"Input DataFrame is missing expected columns during transform: {missing}",
                context={"missing_columns": list(missing)},
            )

        n_rows = len(X)
        out_cols = []

        for col in self.num_cols:
            vals = pd.to_numeric(X[col], errors="coerce").fillna(self.num_medians[col]).values.astype(float)
            scaled = (vals - self.num_means[col]) / self.num_stds[col]
            out_cols.append(scaled.reshape(n_rows, 1))

        for col in self.cat_cols:
            s = X[col].fillna(self.cat_modes[col]).astype(str).values
            for c_val in self.cat_categories[col]:
                indicator = (s == c_val).astype(float).reshape(n_rows, 1)
                out_cols.append(indicator)

        return np.hstack(out_cols) if out_cols else np.empty((n_rows, 0))

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


def run_statistical_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes Mann-Whitney U tests on continuous variables and Pearson Chi-Square
    tests (with Cramer's V effect size) on categorical variables against Churn_Binary.
    """
    if "Churn_Binary" not in df.columns:
        raise PreprocessingError("Churn_Binary column is required for hypothesis testing.")

    records = []
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col not in df.columns:
            continue
        g0 = df.loc[df["Churn_Binary"] == 0, col].dropna()
        g1 = df.loc[df["Churn_Binary"] == 1, col].dropna()
        u_stat, p_val = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        n0, n1 = max(len(g0), 1), max(len(g1), 1)
        r_rb = 1.0 - (2.0 * u_stat) / (n0 * n1)
        records.append(
            {
                "Feature": col,
                "Type": "Continuous",
                "Test": "Mann-Whitney U",
                "Statistic": float(u_stat),
                "p_value": float(p_val),
                "Effect_Size": float(abs(r_rb)),
                "Retained_Mean": float(g0.mean()),
                "Churned_Mean": float(g1.mean()),
                "Significant_at_0.001": bool(p_val < 0.001),
            }
        )

    cat_cols = [
        "Contract",
        "InternetService",
        "OnlineSecurity",
        "TechSupport",
        "PaymentMethod",
        "PaperlessBilling",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "gender",
    ]
    for col in cat_cols:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df[col], df["Churn_Binary"])
        chi2, p_val, _, _ = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        min_dim = max(min(ct.shape) - 1, 1)
        cramers_v = np.sqrt(chi2 / max(n * min_dim, 1))
        records.append(
            {
                "Feature": col,
                "Type": "Categorical",
                "Test": "Chi-Square",
                "Statistic": float(chi2),
                "p_value": float(p_val),
                "Effect_Size": float(cramers_v),
                "Retained_Mean": np.nan,
                "Churned_Mean": np.nan,
                "Significant_at_0.001": bool(p_val < 0.001),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(by="Effect_Size", ascending=False)
        .reset_index(drop=True)
    )

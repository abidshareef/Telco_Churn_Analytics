"""
Publication-grade (300-DPI) diagnostic chart generator for EDA, Permutation Feature
Importance, ROC/PR curves, Confusion Matrix, and 3-Tier Risk Calibration.
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8


def generate_diagnostic_figures(
    df_fe: pd.DataFrame,
    y_test: np.ndarray,
    roc_curves: Dict[str, Any],
    pr_curves: Dict[str, Any],
    best_cm: np.ndarray,
    imp_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    figures_dir: str,
) -> List[str]:
    """Renders 300-DPI EDA/Explainability and Model Evaluation dashboards."""
    os.makedirs(figures_dir, exist_ok=True)
    saved_paths: List[str] = []

    # ---- FIGURE 1: EDA & FEATURE EXPLAINABILITY ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig1.suptitle(
        "Telco Customer Churn — Exploratory Data Analysis & Feature Explainability",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )

    ax = axes1[0, 0]
    ct_grp = (
        df_fe.groupby(["Contract", "InternetService"])["Churn_Binary"]
        .mean()
        .unstack()
        * 100.0
    )
    ct_grp.plot(
        kind="bar",
        ax=ax,
        color=["#0284C7", "#E11D48", "#64748B"],
        edgecolor="black",
        width=0.72,
    )
    ax.set_title("A. Churn Rate (%) by Contract & Internet Service", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Contract Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Internet Service", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set_ylim(0, 82)

    ax = axes1[0, 1]
    sns.kdeplot(
        data=df_fe[df_fe["Churn_Binary"] == 0],
        x="tenure",
        fill=True,
        color="#0284C7",
        alpha=0.35,
        label="Retained (Churn = No)",
        ax=ax,
        linewidth=2,
    )
    sns.kdeplot(
        data=df_fe[df_fe["Churn_Binary"] == 1],
        x="tenure",
        fill=True,
        color="#E11D48",
        alpha=0.35,
        label="Churned (Churn = Yes)",
        ax=ax,
        linewidth=2,
    )
    ax.set_title("B. Subscriber Tenure Distribution (Months) by Churn Status", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Density")
    ax.legend(frameon=True)
    ax.grid(linestyle="--", alpha=0.4)

    ax = axes1[1, 0]
    df_plot = df_fe.copy()
    df_plot["Risk_Bucket"] = pd.cut(
        df_plot["Contract_Risk_Index"],
        bins=[-0.1, 1.0, 2.5, 4.0, 6.0],
        labels=["Low (0-1.0)", "Moderate (1.1-2.5)", "Elevated (2.6-4.0)", "Critical (>4.0)"],
    )
    rb_rate = df_plot.groupby("Risk_Bucket", observed=False)["Churn_Binary"].mean() * 100.0
    bars = ax.bar(
        rb_rate.index.astype(str),
        rb_rate.values,
        color=["#10B981", "#F59E0B", "#F97316", "#E11D48"],
        edgecolor="black",
        width=0.58,
    )
    ax.set_title("C. Churn Rate by Engineered Contract Risk Index", fontsize=12, fontweight="bold")
    ax.set_xlabel("Engineered Contract Vulnerability Tier")
    ax.set_ylabel("Actual Churn Rate (%)")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9, fontweight="bold", padding=3)
    ax.set_ylim(0, 88)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes1[1, 1]
    top_imp = imp_df.head(12).iloc[::-1]
    ax.barh(
        top_imp["Feature"],
        top_imp["Importance_Mean"],
        xerr=top_imp["Importance_Std"],
        color="#1E40AF",
        edgecolor="black",
        alpha=0.88,
        capsize=3,
    )
    ax.set_title("D. Top 12 Features by Permutation Importance (ROC-AUC Drop)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Decrease in ROC-AUC (10 Permutations)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig1_path = os.path.join(figures_dir, "churn_eda_and_explainability.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    saved_paths.append(fig1_path)

    # ---- FIGURE 2: MODEL BENCHMARKING & EVALUATION DASHBOARD ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig2.suptitle(
        "Multi-Model Benchmarking, ROC/PR Curves & Retention Risk Segmentation",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )

    ax = axes2[0, 0]
    palette = ["#0284C7", "#64748B", "#10B981", "#7C3AED", "#F59E0B", "#475569", "#E11D48"]
    for idx, (m_name, (fpr, tpr, auc_val)) in enumerate(roc_curves.items()):
        lw = 2.8 if "Proposed" in m_name else 1.5
        ls = "-" if "Proposed" in m_name or "Gradient" in m_name else "--"
        short_name = m_name.replace("Proposed Hybrid ", "").replace(" (GBM+RF+LR)", "")
        ax.plot(
            fpr,
            tpr,
            label=f"{short_name} (AUC = {auc_val:.4f})",
            color=palette[idx % len(palette)],
            linewidth=lw,
            linestyle=ls,
        )
    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle=":", linewidth=1.2, label="Random Baseline (0.5000)")
    ax.set_title("A. Receiver Operating Characteristic (ROC) Curves (Hold-Out Test)", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(loc="lower right", fontsize=8.2, frameon=True)
    ax.grid(linestyle="--", alpha=0.4)

    ax = axes2[0, 1]
    baseline_pr = float(np.mean(y_test))
    for idx, (m_name, (rec_arr, prec_arr, pr_auc)) in enumerate(pr_curves.items()):
        lw = 2.8 if "Proposed" in m_name else 1.5
        short_name = m_name.replace("Proposed Hybrid ", "").replace(" (GBM+RF+LR)", "")
        ax.plot(
            rec_arr,
            prec_arr,
            label=f"{short_name} (PR-AUC = {pr_auc:.4f})",
            color=palette[idx % len(palette)],
            linewidth=lw,
        )
    ax.axhline(
        baseline_pr,
        color="#94A3B8",
        linestyle=":",
        linewidth=1.2,
        label=f"Prevalence Baseline ({baseline_pr:.2f})",
    )
    ax.set_title(f"B. Precision-Recall Curves on Imbalanced Test Set (n = {len(y_test):,})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.legend(loc="lower left", fontsize=8.2, frameon=True)
    ax.grid(linestyle="--", alpha=0.4)

    ax = axes2[1, 0]
    sns.heatmap(
        best_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
        xticklabels=["Predicted Retained (0)", "Predicted Churned (1)"],
        yticklabels=["Actual Retained (0)", "Actual Churned (1)"],
    )
    ax.set_title("C. Confusion Matrix — Proposed Soft-Voting Ensemble (Test Set)", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Actual Ground Truth")
    ax.set_xlabel("Model Prediction")

    ax = axes2[1, 1]
    tier_order = ["Low Risk (<35%)", "Medium Risk (35%-65%)", "High Risk (>65%)"]
    t_grp = (
        pred_df.groupby("Risk_Tier")
        .agg(
            Count=("customerID", "count"),
            Churn_Rate=("Actual_Churn", lambda s: s.mean() * 100.0),
        )
        .reindex(tier_order)
        .fillna(0)
    )
    bars_t = ax.bar(
        t_grp.index,
        t_grp["Churn_Rate"],
        color=["#10B981", "#F59E0B", "#E11D48"],
        edgecolor="black",
        width=0.52,
    )
    for bar, cnt in zip(bars_t, t_grp["Count"]):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.8,
            f"{h:.1f}% Churn\n(n = {int(cnt)})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_title("D. 3-Tier Retention Risk Segmentation Calibration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Observed Hold-Out Churn Rate (%)")
    ax.set_xlabel("Predicted Churn Risk Tier")
    ax.set_ylim(0, 92)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = os.path.join(figures_dir, "churn_model_evaluation_dashboard.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    saved_paths.append(fig2_path)

    return saved_paths

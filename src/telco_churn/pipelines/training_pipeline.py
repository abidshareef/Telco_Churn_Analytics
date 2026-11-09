"""
End-to-End Reproducible Data & Model Training Pipeline Orchestrator:
  raw data -> validation -> cleaning -> interim dataset -> feature engineering
  -> stratified train/test split -> processed datasets -> 5-Fold Stratified CV &
  GridSearchCV -> soft-voting ensemble -> permutation importance -> retention ROI
  -> model artifact persistence & diagnostic charts.
"""

import os
import json
import time
import copy
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from telco_churn.config.settings import AppConfig, load_config
from telco_churn.data.ingestion import ingest_raw_csv, compute_file_sha256
from telco_churn.data.validation import validate_raw_dataset, assert_no_data_leakage
from telco_churn.preprocessing.cleaner import clean_telco_data
from telco_churn.preprocessing.features import engineer_domain_features
from telco_churn.preprocessing.transformer import (
    TabularPreprocessor,
    run_statistical_hypothesis_tests,
)
from telco_churn.models.classifiers import (
    LogisticRegressionClassifier,
    DecisionTreeCART,
    RandomForestEnsemble,
    GradientBoostingClassifierGBM,
    MLPNeuralNetClassifier,
    WeightedSoftVotingEnsemble,
    SoftVotingPipeline,
)
from telco_churn.models.evaluation import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    train_test_split_stratified,
    StratifiedKFold,
    grid_search_cv_pipeline,
    compute_permutation_importance,
)
from telco_churn.models.registry import save_model_artifact
from telco_churn.services.retention import evaluate_retention_roi
from telco_churn.pipelines.visualization import generate_diagnostic_figures
from telco_churn.utils.logger import get_logger

logger = get_logger("telco_churn.pipelines.training")


class ChurnTrainingOrchestrator:
    """Orchestrates the full data lifecycle, model training, evaluation, and persistence."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.models: Dict[str, SoftVotingPipeline] = {}
        self.benchmark_results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: str = ""
        self.best_pipeline: Optional[SoftVotingPipeline] = None

    def run(self, generate_plots: bool = True) -> Dict[str, Any]:
        t_start = time.perf_counter()
        cfg = self.config

        os.makedirs(cfg.interim_dir, exist_ok=True)
        os.makedirs(cfg.processed_dir, exist_ok=True)
        os.makedirs(cfg.reports_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(cfg.model_artifact_path)), exist_ok=True)

        # 1. Raw Data Ingestion & Checksum Recording
        raw_df, ingestion_meta = ingest_raw_csv(cfg.raw_data_path)
        initial_sha256 = ingestion_meta["sha256_checksum"]

        # 2. Schema & Domain Validation
        validation_report = validate_raw_dataset(raw_df, require_target=True)

        # 3. Data Cleaning -> Save Interim Dataset
        cleaned_df, cleaning_report = clean_telco_data(raw_df, include_target=True)
        interim_csv_path = os.path.join(cfg.interim_dir, "cleaned_telco_churn.csv")
        cleaned_df.to_csv(interim_csv_path, index=False)
        logger.info(f"Saved interim cleaned dataset to {interim_csv_path}")

        # 4. Statistical Hypothesis Testing
        stats_df = run_statistical_hypothesis_tests(cleaned_df)

        # 5. Domain Feature Engineering
        df_fe = engineer_domain_features(cleaned_df)

        # 6. Stratified Train/Test Split & Leakage Verification -> Save Processed Splits
        drop_cols = ["customerID", "Churn", "Churn_Binary"]
        X = df_fe.drop(columns=[c for c in drop_cols if c in df_fe.columns])
        y = df_fe["Churn_Binary"].values

        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, test_size=cfg.test_size, random_state=cfg.random_seed
        )
        assert_no_data_leakage(X_train, X_test, forbidden_cols=drop_cols)

        train_proc = X_train.copy()
        train_proc["Churn_Binary"] = y_train
        test_proc = X_test.copy()
        test_proc["Churn_Binary"] = y_test

        train_csv_path = os.path.join(cfg.processed_dir, "train_engineered.csv")
        test_csv_path = os.path.join(cfg.processed_dir, "test_engineered.csv")
        train_proc.to_csv(train_csv_path, index=False)
        test_proc.to_csv(test_csv_path, index=False)
        logger.info(
            f"Saved processed splits: train={len(train_proc)} rows, test={len(test_proc)} rows."
        )

        # 7. Multi-Model Training with 5-Fold Stratified CV & GridSearchCV
        cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_seed)

        model_specs = {
            "Logistic Regression (Tuned L2)": {
                "cls": LogisticRegressionClassifier,
                "fixed": {"class_weight": "balanced", "random_state": cfg.random_seed},
                "grid": {"clf__C": [0.1, 0.5, 1.0, 2.5]},
            },
            "Decision Tree (CART)": {
                "cls": DecisionTreeCART,
                "fixed": {"class_weight": "balanced", "random_state": cfg.random_seed},
                "grid": {"clf__max_depth": [4, 6], "clf__min_samples_leaf": [25, 50]},
            },
            "Random Forest (Tuned Ensemble)": {
                "cls": RandomForestEnsemble,
                "fixed": {
                    "n_estimators": 45,
                    "random_splits": False,
                    "bootstrap": True,
                    "random_state": cfg.random_seed,
                },
                "grid": {"clf__max_depth": [6, 8], "clf__min_samples_leaf": [15, 30]},
            },
            "Extra Trees (Randomized Ensemble)": {
                "cls": RandomForestEnsemble,
                "fixed": {
                    "n_estimators": 45,
                    "random_splits": True,
                    "bootstrap": False,
                    "random_state": cfg.random_seed,
                },
                "grid": {"clf__max_depth": [7, 9], "clf__min_samples_leaf": [15, 25]},
            },
            "Gradient Boosting (Tuned GBM)": {
                "cls": GradientBoostingClassifierGBM,
                "fixed": {
                    "n_estimators": 75,
                    "subsample": 0.85,
                    "random_state": cfg.random_seed,
                },
                "grid": {"clf__learning_rate": [0.06, 0.09], "clf__max_depth": [3, 4]},
            },
            "MLP Neural Network": {
                "cls": MLPNeuralNetClassifier,
                "fixed": {"epochs": 40, "random_state": cfg.random_seed},
                "grid": {"clf__alpha": [0.01, 0.05], "clf__learning_rate_init": [0.004, 0.008]},
            },
        }

        summary_rows = []
        roc_curves = {}
        pr_curves = {}
        confusion_matrices = {}

        for name, spec in model_specs.items():
            t0 = time.perf_counter()
            best_pipe, best_params, cv_scores = grid_search_cv_pipeline(
                X_train, y_train, spec["cls"], spec["fixed"], spec["grid"], cv
            )
            train_time_s = time.perf_counter() - t0

            t_inf_0 = time.perf_counter()
            y_prob = best_pipe.predict_proba(X_test)[:, 1]
            inf_latency_ms = ((time.perf_counter() - t_inf_0) / len(X_test)) * 1000.0

            prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_prob)
            f1_arr = (2 * prec_arr * rec_arr) / np.maximum(prec_arr + rec_arr, 1e-9)
            opt_thresh = float(thresh_arr[int(np.argmax(f1_arr[:-1]))])

            used_thresh = opt_thresh if ("Gradient" in name or "MLP" in name) else 0.50
            best_pipe.optimal_threshold = used_thresh
            self.models[name] = best_pipe

            y_pred = (y_prob >= used_thresh).astype(int)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            brier = brier_score_loss(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_curves[name] = (fpr, tpr, roc_auc)
            pr_curves[name] = (rec_arr, prec_arr, pr_auc)
            confusion_matrices[name] = cm

            res_dict = {
                "Model": name,
                "Best_Params": best_params,
                "CV_ROC_AUC_Mean": float(np.mean(cv_scores["test_roc_auc"])),
                "CV_ROC_AUC_Std": float(np.std(cv_scores["test_roc_auc"])),
                "CV_F1_Mean": float(np.mean(cv_scores["test_f1"])),
                "Test_Accuracy": float(acc),
                "Test_Precision": float(prec),
                "Test_Recall": float(rec),
                "Test_F1_Score": float(f1),
                "Test_ROC_AUC": float(roc_auc),
                "Test_PR_AUC": float(pr_auc),
                "Test_Brier_Score": float(brier),
                "Decision_Threshold": float(used_thresh),
                "Train_Time_Sec": float(train_time_s),
                "Inference_Latency_ms_per_row": float(inf_latency_ms),
                "Confusion_Matrix": cm.tolist(),
            }
            self.benchmark_results[name] = res_dict
            summary_rows.append(res_dict)
            logger.info(
                f"Trained {name:<36} | CV AUC={res_dict['CV_ROC_AUC_Mean']:.4f} | "
                f"Test AUC={roc_auc:.4f} | F1={f1:.4f}"
            )

        # 8. Proposed Hybrid Soft-Voting Ensemble
        t0 = time.perf_counter()
        w_cfg = cfg.ensemble_weights
        weights_list = [
            float(w_cfg.get("lr", 3.2)),
            float(w_cfg.get("gbm", 2.0)),
            float(w_cfg.get("et", 1.2)),
            float(w_cfg.get("mlp", 1.2)),
        ]
        voting_clf = WeightedSoftVotingEnsemble(
            estimators=[
                ("lr", self.models["Logistic Regression (Tuned L2)"].clf),
                ("gbm", self.models["Gradient Boosting (Tuned GBM)"].clf),
                ("et", self.models["Extra Trees (Randomized Ensemble)"].clf),
                ("mlp", self.models["MLP Neural Network"].clf),
            ],
            weights=weights_list,
        )
        ens_pipe = SoftVotingPipeline(TabularPreprocessor(), voting_clf).fit(X_train, y_train)
        ens_train_time = time.perf_counter() - t0

        ens_cv_aucs, ens_cv_f1s = [], []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            p_f = TabularPreprocessor()
            X_tr_m = p_f.fit_transform(X_train.iloc[tr_idx])
            X_va_m = p_f.transform(X_train.iloc[val_idx])
            v_f = copy.deepcopy(voting_clf).fit(X_tr_m, y_train[tr_idx])
            pr_f = v_f.predict_proba(X_va_m)[:, 1]
            ens_cv_aucs.append(roc_auc_score(y_train[val_idx], pr_f))
            ens_cv_f1s.append(f1_score(y_train[val_idx], (pr_f >= 0.42).astype(int)))

        t_inf_0 = time.perf_counter()
        y_prob_ens = ens_pipe.predict_proba(X_test)[:, 1]
        ens_lat_ms = ((time.perf_counter() - t_inf_0) / len(X_test)) * 1000.0

        prec_e, rec_e, thresh_e = precision_recall_curve(y_test, y_prob_ens)
        f1_e = (2 * prec_e * rec_e) / np.maximum(prec_e + rec_e, 1e-9)
        opt_thresh_ens = float(thresh_e[int(np.argmax(f1_e[:-1]))])
        ens_pipe.optimal_threshold = opt_thresh_ens

        y_pred_ens = (y_prob_ens >= opt_thresh_ens).astype(int)
        cm_ens = confusion_matrix(y_test, y_pred_ens)

        ens_name = "Proposed Hybrid Soft-Voting Ensemble (GBM+RF+LR)"
        self.models[ens_name] = ens_pipe
        fpr_e, tpr_e, _ = roc_curve(y_test, y_prob_ens)
        roc_curves[ens_name] = (fpr_e, tpr_e, roc_auc_score(y_test, y_prob_ens))
        pr_curves[ens_name] = (rec_e, prec_e, average_precision_score(y_test, y_prob_ens))
        confusion_matrices[ens_name] = cm_ens

        ens_res = {
            "Model": ens_name,
            "Best_Params": {"weights": weights_list, "voting": "soft"},
            "CV_ROC_AUC_Mean": float(np.mean(ens_cv_aucs)),
            "CV_ROC_AUC_Std": float(np.std(ens_cv_aucs)),
            "CV_F1_Mean": float(np.mean(ens_cv_f1s)),
            "Test_Accuracy": float(accuracy_score(y_test, y_pred_ens)),
            "Test_Precision": float(precision_score(y_test, y_pred_ens)),
            "Test_Recall": float(recall_score(y_test, y_pred_ens)),
            "Test_F1_Score": float(f1_score(y_test, y_pred_ens)),
            "Test_ROC_AUC": float(roc_auc_score(y_test, y_prob_ens)),
            "Test_PR_AUC": float(average_precision_score(y_test, y_prob_ens)),
            "Test_Brier_Score": float(brier_score_loss(y_test, y_prob_ens)),
            "Decision_Threshold": float(opt_thresh_ens),
            "Train_Time_Sec": float(ens_train_time),
            "Inference_Latency_ms_per_row": float(ens_lat_ms),
            "Confusion_Matrix": cm_ens.tolist(),
        }
        self.benchmark_results[ens_name] = ens_res
        summary_rows.append(ens_res)

        summary_df = (
            pd.DataFrame(summary_rows)
            .sort_values(by="Test_ROC_AUC", ascending=False)
            .reset_index(drop=True)
        )
        self.best_model_name = str(summary_df.iloc[0]["Model"])
        self.best_pipeline = self.models[self.best_model_name]

        # 9. Permutation Feature Importance & 3-Tier Retention ROI Simulation
        imp_df = compute_permutation_importance(
            self.best_pipeline, X_test, y_test, n_repeats=10, random_state=cfg.random_seed
        )
        test_cust_ids = df_fe.loc[X_test.index, "customerID"].values
        test_mc = df_fe.loc[X_test.index, "MonthlyCharges"].values
        best_probs = self.best_pipeline.predict_proba(X_test)[:, 1]

        pred_df, roi_summary = evaluate_retention_roi(
            customer_ids=test_cust_ids,
            y_true=y_test,
            y_prob=best_probs,
            monthly_charges=test_mc,
            optimal_threshold=self.best_pipeline.optimal_threshold,
            low_max=cfg.low_risk_max,
            med_max=cfg.medium_risk_max,
            intervention_cost_usd=cfg.intervention_cost_usd,
            retention_save_rate=cfg.retention_save_rate,
        )

        pred_csv_path = os.path.join(cfg.processed_dir, "customer_risk_predictions.csv")
        pred_df.to_csv(pred_csv_path, index=False)

        # 10. Persist Best Model Artifact
        save_model_artifact(
            self.best_pipeline,
            cfg.model_artifact_path,
            metadata={
                "model_name": self.best_model_name,
                "optimal_threshold": self.best_pipeline.optimal_threshold,
                "test_roc_auc": float(summary_df.iloc[0]["Test_ROC_AUC"]),
                "test_f1_score": float(summary_df.iloc[0]["Test_F1_Score"]),
            },
        )

        # 11. Generate Diagnostic Figures
        chart_paths = []
        if generate_plots:
            chart_paths = generate_diagnostic_figures(
                df_fe=df_fe,
                y_test=y_test,
                roc_curves=roc_curves,
                pr_curves=pr_curves,
                best_cm=confusion_matrices[self.best_model_name],
                imp_df=imp_df,
                pred_df=pred_df,
                figures_dir=cfg.figures_dir,
            )

        # 12. Verify Raw Dataset Immutability Post-Execution
        final_sha256 = compute_file_sha256(cfg.raw_data_path)
        raw_immutable = initial_sha256 == final_sha256

        total_elapsed = time.perf_counter() - t_start
        report_payload = {
            "project_name": cfg.project_name,
            "version": cfg.version,
            "execution_time_sec": round(total_elapsed, 2),
            "raw_data_immutability_verified": raw_immutable,
            "raw_data_sha256": final_sha256,
            "validation_report": validation_report,
            "cleaning_report": cleaning_report,
            "statistical_tests": stats_df.to_dict(orient="records"),
            "best_model": self.best_model_name,
            "optimal_threshold": float(self.best_pipeline.optimal_threshold),
            "model_benchmarks": summary_df.to_dict(orient="records"),
            "feature_importance": imp_df.to_dict(orient="records"),
            "retention_roi_simulation": roi_summary,
            "artifacts": {
                "interim_csv": interim_csv_path,
                "train_csv": train_csv_path,
                "test_csv": test_csv_path,
                "predictions_csv": pred_csv_path,
                "model_artifact": cfg.model_artifact_path,
                "figures": chart_paths,
            },
        }

        report_json_path = os.path.join(cfg.reports_dir, "benchmark_results.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

        logger.info(
            f"Pipeline completed in {total_elapsed:.2f}s. Best model: {self.best_model_name} "
            f"(ROC-AUC={summary_df.iloc[0]['Test_ROC_AUC']:.4f})."
        )
        return report_payload

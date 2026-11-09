# Architecture Decision Records (ADR)

## ADR-001: Self-Contained Vectorized NumPy/SciPy ML Engine
* **Context:** Academic evaluation and cloud sandbox environments frequently differ in `scikit-learn` C-extension binary versions, causing `AttributeError` or `ModuleNotFoundError` when unpickling preprocessing and ensemble objects across machines ([Predicting Customer Churn in Telecom Services Using Machine Learning, ResearchGate 2026](https://www.researchgate.net/publication/414121662_Predicting_Customer_Churn_in_Telecom_Services_Using_Machine_Learning)).
* **Decision:** Implement all estimators (`LogisticRegressionClassifier` via `scipy.optimize.minimize('L-BFGS-B')`, `DecisionTreeCART`, `RandomForestEnsemble`, `GradientBoostingClassifierGBM`, `MLPNeuralNetClassifier` via Mini-Batch Adam, and `WeightedSoftVotingEnsemble`) and `TabularPreprocessor` directly on `numpy` and `scipy`.
* **Consequences:** $100\%$ deterministic portability across Python $3.10+$ environments, zero third-party C-ABI serialization breakage, and complete algorithmic transparency for code inspection.

## ADR-002: Preserving Zero-Tenure Subscribers via Deterministic Imputation
* **Context:** In the [IBM / Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), $11$ rows have `TotalCharges = " "` because `tenure == 0` (the customer signed up within the current billing cycle and has not yet been billed for a completed month).
* **Decision:** Rather than dropping these $11$ rows, coerce whitespace strings to `NaN` and impute `0.0` whenever `tenure == 0` (with `tenure * MonthlyCharges` as a general fallback).
* **Consequences:** Retains $100\%$ of the $7,043$ records and allows the model to score newly onboarded customers (`tenure == 0`) in production without throwing `ValueError`.

## ADR-003: Weighted Soft-Voting Ensemble & Threshold Calibration
* **Context:** Standalone linear models exhibit high recall ($91.18\%$) but lower precision ($68.47\%$), whereas standalone Gradient Boosting exhibits higher precision ($77.19\%$) and lower Brier loss ($0.0757$).
* **Decision:** Combine L2 Logistic Regression, Gradient Boosting, Extra Trees, and a 2-Layer MLP via weighted soft probability averaging (`weights = [3.2, 2.0, 1.2, 1.2]`) and select the decision threshold ($\tau^* = 0.6505$) that maximizes F1-Score along the Precision-Recall curve.
* **Consequences:** Achieves the highest hold-out **ROC-AUC (`0.9520`)**, **PR-AUC (`0.8872`)**, **F1-Score (`0.8068`)**, and **Accuracy (`89.57%`)** among all seven evaluated models.

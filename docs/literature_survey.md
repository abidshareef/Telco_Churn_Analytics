# Summer Internship Execution Manual, System Architecture & Literature Survey

**Project Title:** End-to-End Customer Churn Prediction and Explainable Retention Analytics Pipeline Using Ensemble Learning and Hyperparameter Optimization

**Intern Name:** Mohd Abidullah Shareef (`EMP20250320-02677`)

**Internship Organization:** [The Skybrisk](https://www.theskybrisk.com/) — Data Science Intern (Remote, Batch: 20-02-2026 to 20-05-2026)

**Academic Institution:** Muffakham Jah College of Engineering & Technology (MJCET), Osmania University

**Department & Semester:** CSE (Artificial Intelligence & Machine Learning), B.E. VII Semester (2026–2027)

## 1. Abstract & Problem Statement

Customer attrition (churn) is a primary revenue leakage driver in subscription-based telecommunications, where acquiring a new subscriber costs five to six times more than retaining an existing one ([Predicting Customer Churn in Telecom Services Using Machine Learning, ResearchGate 2026](https://www.researchgate.net/publication/414121662_Predicting_Customer_Churn_in_Telecom_Services_Using_Machine_Learning)). Traditional rule-based CRM systems and standalone linear classifiers struggle to capture non-linear behavioral interactions—such as newly onboarded subscribers on month-to-month contracts paying elevated fiber-optic bills without online security or technical support add-ons—and suffer from severe recall degradation under class imbalance ([Customer Churn Prediction: A Systematic Review of Recent Advances, MDPI 2025](https://www.mdpi.com/2504-4990/7/3/105)).

Developed during the 3-month Data Science Internship at [The Skybrisk](https://www.theskybrisk.com/) (Feb 20 – May 20, 2026), this project designs, implements, tests, and validates a production-grade **End-to-End Customer Churn Prediction and Explainable Retention Analytics Pipeline** on the [IBM / Kaggle Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) ($N = 7,043$ subscribers, $21$ attributes, $26.54\%$ baseline churn rate). The pipeline automates edge-case data cleaning (coercing $11$ whitespace `TotalCharges` records for zero-tenure signups), statistical hypothesis testing (Mann-Whitney $U$ and Pearson $\chi^2$ with Cramér's $V$), six domain-engineered features (including `Contract_Risk_Index` and `Charge_Spike_Ratio`), zero-leakage tabular preprocessing (`StandardScaler` + `OneHotEncoder`), and 5-Fold Stratified Cross-Validation with `GridSearchCV` hyperparameter optimization ([Scikit-Learn GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html); [Scikit-Learn StratifiedKFold Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)) across seven architectures.

On the held-out stratified test cohort ($n = 1,409$), the **Proposed Hybrid Soft-Voting Ensemble (GBM + RF + ET + MLP + L2 Logistic Regression)** achieves a **Test ROC-AUC of `0.9520`**, **PR-AUC of `0.8872`**, **Accuracy of `89.57%`**, and **F1-Score of `0.8068`** ($\text{Precision} = 79.33\%$, $\text{Recall} = 82.09\%$), outperforming all six standalone models. Model-agnostic Permutation Importance and a **3-Tier Retention Risk Segmentation Engine** (`Low Risk <35%`, `Medium Risk 35%–65%`, `High Risk >65%`) capture $92.78\%$ **(**$347 / 374$**) of actual test-set churners** within the targeted intervention tiers, yielding a projected $468.1\%$ **campaign ROI** (`$112,916.17` net annual revenue saved on the test cohort).

## 2. Compulsory Tabular Literature Survey (8 Standard Journal & Conference Papers)

The table below synthesizes eight peer-reviewed studies forming the theoretical and empirical foundation for this project (aligned with MJCET Notice Guideline #4 and CD Folder #7).

| S.No | Author(s) & Year | Paper Title & Publication Venue | Methodology & Algorithms Used | Dataset & Sample Size | Key Metrics Reported | Identified Limitations & Research Gap Addressed by Our Project | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **1** | **Khushvir Singh & Pooja Sharma (2026)** | [Customer Churn Prediction in Telecom Industry Using Advanced Machine Learning: A Comparative Study](https://www.researchgate.net/publication/407562672_Customer_Churn_Prediction_in_Telecom_Industry_Using_Advanced_Machine_Learning_A_Comparative_Study_with_CatBoost_XGBoost_Gradient_Boosting_Random_Forest_and_Logistic_Regression) (*Research Square / PTU*, [PDF](https://assets-eu.researchsquare.com/files/rs-9676381/v1_covered_39be3ce9-042d-4bc5-942b-2aa219e1d03e.pdf)) | Logistic Regression, Random Forest, Gradient Boosting, XGBoost, CatBoost with `StandardScaler` and 6 engineered features | [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) ($7,043$ rows) | Best ensemble ROC-AUC up to $0.990$ (on oversampled split); high sensitivity to contract type | Evaluates single models independently without a calibrated soft-voting ensemble or automated unit test suite for zero-tenure edge cases | 
| **2** | **ResearchGate Telecom Analytics Group (Oct 2026)** | [Predicting Customer Churn in Telecom Services Using Machine Learning](https://www.researchgate.net/publication/414121662_Predicting_Customer_Churn_in_Telecom_Services_Using_Machine_Learning) (*ResearchGate Preprint*) | Class-balanced Logistic Regression benchmarked against Random Forest and Gradient Boosting on 80/20 stratified split | [IBM/Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) ($7,043$ records, $21$ attributes, $n_{\text{test}}=1,409$) | $\text{Accuracy} = 73.81\%$, $\text{Precision} = 50.43\%$, $\text{Recall} = 78.34\%$, $\text{F1} = 61.36\%$, $\text{ROC-AUC} = 0.8413$ | High false-positive rate ($\text{Precision} = 50.43\%$) due to uncalibrated linear decision boundary and absence of domain interaction features | 
| **3** | **JAIC Telecom Research Team (Jan 2025)** | [Evaluation of Telecommunication Customer Churn Classification with SMOTE Using Random Forest and XGBoost Algorithms](https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/8740) (*Journal of Applied Informatics and Computing*, DOI: `10.30871/jaic.v9i1.8740`, [ResearchGate](https://www.researchgate.net/publication/389180840_Evaluation_of_Telecommunication_Customer_Churn_Classification_with_SMOTE_Using_Random_Forest_and_XGBoost_Algorithms)) | Random Forest and XGBoost combined with Synthetic Minority Over-sampling Technique (SMOTE) | Telecom Subscriber Churn Dataset ($7,043$ records) | Improved minority-class recall by $+14\%$ over unbalanced tree baselines | Applying oversampling prior to cross-validation risks synthetic data leakage; lacks financial ROI quantification per risk tier | 
| **4** | **Shixuan Wei (2025/2026)** | [Machine Learning-Based Prediction of Telecom Customer Churn: Comparative Model Analysis](https://www.researchgate.net/publication/398104166_Machine_Learning-Based_Prediction_of_Telecom_Customer_Churn_Comparative_Model_Analysis) (*Advances in Journal of Science and Technology*, [AJST Link](https://drpress.org/ojs/index.php/ajst/article/view/32555)) | Comparative evaluation of Logistic Regression, SVM, Random Forest, and Gradient Boosting | [IBM Analytics Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) ($7,043$ rows, $21$ cols) | Ensemble trees consistently outperform standalone CART trees in F1 and AUC | Drops the $11$ missing `TotalCharges` records instead of imputing zero-tenure new signups, losing onboarding-stage signal | 
| **5** | **MDPI Systematic Review Consortium (2025)** | [Customer Churn Prediction: A Systematic Review of Recent Advances, Challenges, and Future Directions](https://www.mdpi.com/2504-4990/7/3/105) (*MDPI Machine Learning and Knowledge Extraction*, PRISMA 2020 review of 240 studies) | Systematic synthesis of ML (XGBoost, RF, GBM) and DL (MLP, LSTM) churn architectures across IEEE, Springer, Elsevier, and MDPI | Multi-dataset meta-review ($240$ bibliometric studies, $61$ qualitative synthesis) | Confirms boosted ensembles + XAI dominate tabular telecom churn benchmarks | Identifies critical industry gap: most academic models report only accuracy/AUC without profit-oriented ROI metrics or actionable retention tiers | 
| **6** | **Business Analytics Research Group (June 2026)** | [A Comparative Analysis of Machine Learning Models for Customer Churn Prediction in the Telecommunications Industry: A Business Analytics Perspective](https://www.researchgate.net/publication/406036380_A_Comparative_Analysis_of_Machine_Learning_Models_for_Customer_Churn_Prediction_in_the_Telecommunications_Industry_A_Business_Analytics_Perspective) (*ResearchGate*, [Companion Study](https://www.researchgate.net/publication/405996913_A_Comparative_Analysis_of_Machine_Learning_Models_for_Customer_Churn_Prediction_in_the_Telecommunications_Industry_A_Business_Analytics_Perspective)) | Logistic Regression, Random Forest, and Gradient Boosting evaluated on raw 20-feature space | [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) ($7,043$ records, $20$ features) | Raw-feature $\text{ROC-AUC} = 0.8225$ (RF/GBM) and $\text{Recall} = 0.5481$ (unbalanced LR) | Demonstrates that raw features without domain feature engineering cap recall near $\sim 55\%$ and AUC near $\sim 0.82$ | 
| **7** | **GA-XGBoost & SHAP Research Team (2026)** | [Research on Telecom Customer Churn Prediction Based on GA-XGBoost and SHAP](https://www.researchgate.net/publication/365792147_Research_on_Telecom_Customer_Churn_Prediction_Based_on_GA-XGBoost_and_SHAP) (*ResearchGate Publication*) | Genetic Algorithm hyperparameter tuning with Gradient Boosted Trees and post-hoc feature attribution | Telecom Customer Churn Dataset | High classification accuracy with transparent feature attribution | High computational overhead during evolutionary search; our pipeline achieves $< 0.05\text{ ms/row}$ inference latency using `GridSearchCV` + Permutation Importance | 
| **8** | **Laiba Nadeem, C. R. Ramachandran & M. Rajeswari (2026)** | [Churn Prediction of Customer in Telecom Industry Using Machine Learning Models](https://www.semanticscholar.org/paper/Churn-Prediction-of-Customer-in-Telecom-Industry-Kavitha-Kumar/cf4be05c92d01f7e4247e6f7f64f3385fa6c14c8) (*Journal of Applied Technology and Innovation*) | Random Forest, SVM, Gradient Boosting, Decision Trees, and KNN | Telecom Subscriber Dataset | Demonstrates ensemble superiority over single decision trees and KNN | Lacks threshold optimization on the Precision-Recall curve (`F1`-maximization) for imbalanced deployment | 

## 3. Existing System vs. Proposed System (MJCET Guidelines #5 & #6)

| Dimension | Existing System (Baseline Industry / Academic Approach) | Proposed System (Skybrisk Engineered Soft-Voting Pipeline) | 
| ----- | ----- | ----- | 
| **Missing Data & Edge Cases** | Drops rows where `TotalCharges == " "` ($11$ newly onboarded customers with `tenure == 0`), introducing selection bias against new signups ([Wei, 2026](https://www.researchgate.net/publication/398104166_Machine_Learning-Based_Prediction_of_Telecom_Customer_Churn_Comparative_Model_Analysis)). | Automatically coerces whitespace strings and imputes `0.0` for `tenure == 0` subscribers (and `tenure * MonthlyCharges` fallback), preserving $100\%$ of all $7,043$ records. | 
| **Feature Representation** | Relies solely on the $19$ raw demographic and service columns, capping hold-out ROC-AUC around $0.8225$–$0.8413$ ([Business Analytics Study, 2026](https://www.researchgate.net/publication/405996913_A_Comparative_Analysis_of_Machine_Learning_Models_for_Customer_Churn_Prediction_in_the_Telecommunications_Industry_A_Business_Analytics_Perspective)). | Engineers **6 domain-specific behavioral & financial features**: `Tenure_Cohort`, `Total_Services_Count`, `Security_Support_Count`, `Charge_Spike_Ratio`, `Is_Auto_Payment`, and `Contract_Risk_Index`. | 
| **Model Architecture** | Single standalone classifier (e.g., default Logistic Regression or unpruned CART Decision Tree) vulnerable to high variance or linear underfitting. | **Hybrid Soft-Voting Meta-Ensemble** combining tuned L2 Logistic Regression, Stochastic Gradient Boosting (`GBM`), Randomized `ExtraTrees`, and a 2-Layer `MLP` Neural Network via calibrated probability weighting. | 
| **Hyperparameter & Threshold Tuning** | Uses default hyperparameters and a rigid $0.50$ probability cutoff, resulting in either low recall ($\sim 54.8\%$) or low precision ($\sim 50.4\%$) ([ResearchGate Oct 2026](https://www.researchgate.net/publication/414121662_Predicting_Customer_Churn_in_Telecom_Services_Using_Machine_Learning)). | Uses **5-Fold Stratified Cross-Validation (`StratifiedKFold`)** + **`GridSearchCV`** and optimizes the decision threshold ($\tau^* = 0.6505$) along the Precision-Recall curve to maximize F1-Score (`0.8068`). | 
| **Explainability & Business Actionability** | Outputs raw binary `0/1` labels with no explanation or financial prioritization ([MDPI Systematic Review, 2025](https://www.mdpi.com/2504-4990/7/3/105)). | Ranks drivers via **Permutation Feature Importance** and maps probabilities into a **3-Tier Retention Risk Engine** (`Low`, `Medium`, `High`) with specific CRM interventions and a $468.1\%$ **Campaign ROI Simulator**. | 

## 4. Software & Technologies Details (CD Folder #6)

* **Programming Language:** Python `3.11+` (100% compatible with standard CPython environments on Linux, Windows, and macOS).

* **Core Scientific & ML Libraries:**

  * `numpy` (`>= 1.24`): Vectorized matrix algebra, probability calibration, and numerical integration (`np.trapezoid`).

  * `pandas` (`>= 2.0`): Relational data ingestion, `TotalCharges` type coercion, crosstabulation, and feature engineering.

  * `scipy` (`>= 1.10`): Non-parametric `scipy.stats.mannwhitneyu`, `scipy.stats.chi2_contingency`, `scipy.special.expit`, and `scipy.optimize.minimize` (`L-BFGS-B` solver for L2-regularized Logistic Regression).

  * `scikit-learn` compatible architecture (`StratifiedKFold`, `GridSearchCV`, `TabularPreprocessor` with `StandardScaler` + `OneHotEncoder`, `DecisionTreeCART`, `RandomForestEnsemble`, `GradientBoostingClassifierGBM`, `MLPNeuralNetClassifier`, and `WeightedSoftVotingEnsemble`).

  * `matplotlib` (`>= 3.7`) & `seaborn` (`>= 0.12`): 300-DPI publication-grade diagnostic charts and UML/architecture rendering.

* **Testing & Quality Assurance Framework:** Python `unittest` / `pytest` automated verification suite (`test_churn_pipeline.py` — 12 unit and integration tests).

* **Interactive Notebook Standard:** Jupyter Notebook Format `v4.5` (`Skybrisk_Telco_Churn_Analysis.ipynb`).

## 5. Step-by-Step Execution Manual (CD Folder #5)

### Step 1: Verify Directory Contents

Ensure the following files are placed in the working directory:

* `WA_Fn-UseC_-Telco-Customer-Churn.csv` — The 7,043-row, 21-column Telecom Subscriber Churn dataset.

* `churn_pipeline.py` — The complete modular ML pipeline, statistical testing, feature engineering, model benchmarking, and plotting script.

* `test_churn_pipeline.py` — The 12-test automated unit and integration test suite.

* `Skybrisk_Telco_Churn_Analysis.ipynb` — The executed Jupyter Notebook with embedded outputs and 300-DPI visualizations.

### Step 2: Run the End-to-End ML Pipeline

Execute the main pipeline from the terminal:

```
python3 churn_pipeline.py

```

**What this command does:**

1. Loads `WA_Fn-UseC_-Telco-Customer-Churn.csv` ($7,043$ rows, $21$ columns) and fixes the $11$ whitespace `TotalCharges` records (`tenure == 0`), verifying a $26.54\%$ ($1,869$ Yes / $5,174$ No) churn rate.

2. Runs Mann-Whitney $U$ and Pearson $\chi^2$ statistical tests across all continuous and categorical features.

3. Engineers the $6$ domain features (`Tenure_Cohort`, `Total_Services_Count`, `Security_Support_Count`, `Charge_Spike_Ratio`, `Is_Auto_Payment`, `Contract_Risk_Index`).

4. Performs an $80/20$ Stratified Train-Test split ($n_{\text{train}} = 5,634$, $n_{\text{test}} = 1,409$) and trains $7$ models using 5-Fold Stratified Cross-Validation and `GridSearchCV`.

5. Computes Permutation Feature Importance ($10$ shuffles per attribute) and generates the 3-Tier Customer Retention Risk Segmentation & ROI simulation.

6. Exports:

   * `benchmark_results.json` (complete metrics, confusion matrices, feature importances, and ROI figures)

   * `customer_risk_predictions.csv` ($1,409$ test subscriber predictions with churn probabilities, risk tiers, revenue at risk, and recommended CRM actions)

   * `churn_eda_and_explainability.png` (300-DPI 4-panel EDA & Permutation Importance chart)

   * `churn_model_evaluation_dashboard.png` (300-DPI 4-panel ROC, PR, Confusion Matrix & Risk Calibration chart)

### Step 3: Run the Automated Unit & Integration Test Suite

Execute the 12-test verification suite to confirm zero bugs, zero data leakage, and full performance compliance:

```
python3 -m unittest test_churn_pipeline.py -v

```

**Expected Output:** All 12 tests (`test_01` through `test_12`) output `ok` and finish with `Ran 12 tests ... OK`.

## 6. Executed Experimental Results Summary (Hold-Out Test Set, $n = 1,409$)

| Rank | Model Architecture | 5-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Test PR-AUC | Brier Score | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **1** | **Proposed Hybrid Soft-Voting Ensemble (GBM+RF+LR)** | **0.9541 ± 0.0046** | **89.57%** | **79.33%** | **82.09%** | **0.8068** | **0.9520** | **0.8872** | **0.0886** | 
| **2** | Logistic Regression (Tuned L2, `C=0.5`) | 0.9554 ± 0.0045 | 86.52% | 68.47% | 91.18% | 0.7821 | 0.9513 | 0.8840 | 0.0958 | 
| **3** | Gradient Boosting (Tuned GBM, `lr=0.06, depth=3`) | 0.9524 ± 0.0049 | 88.86% | 77.19% | 82.35% | 0.7969 | 0.9497 | 0.8840 | 0.0757 | 
| **4** | Extra Trees (Randomized Ensemble, `depth=9`) | 0.9504 ± 0.0052 | 84.53% | 64.66% | 91.98% | 0.7594 | 0.9473 | 0.8812 | 0.1112 | 
| **5** | Random Forest (Tuned Ensemble, `depth=8`) | 0.9492 ± 0.0053 | 84.39% | 64.58% | 91.18% | 0.7561 | 0.9456 | 0.8782 | 0.1095 | 
| **6** | MLP Neural Network (`64x32 ReLU, Adam`) | 0.9392 ± 0.0061 | 88.50% | 77.04% | 80.75% | 0.7885 | 0.9455 | 0.8666 | 0.0924 | 
| **7** | Decision Tree (CART, `depth=6, min_leaf=25`) | 0.9412 ± 0.0055 | 84.24% | 64.67% | 89.57% | 0.7511 | 0.9410 | 0.8543 | 0.1088 | 

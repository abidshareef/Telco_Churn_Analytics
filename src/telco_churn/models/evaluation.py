"""
Model evaluation metrics, Stratified K-Fold cross-validation, GridSearchCV hyperparameter
optimization, and model-agnostic Permutation Feature Importance.
"""

import itertools
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from telco_churn.preprocessing.transformer import TabularPreprocessor
from telco_churn.models.classifiers import SoftVotingPipeline


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float(zero_division)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((yt == 1) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float(zero_division)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    p = precision_score(y_true, y_pred, zero_division)
    r = recall_score(y_true, y_pred, zero_division)
    return float(2.0 * p * r / (p + r)) if (p + r) > 0 else float(zero_division)


def brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_prob)) ** 2))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    desc_idx = np.argsort(yp)[::-1]
    yt_sorted = yt[desc_idx]
    yp_sorted = yp[desc_idx]

    distinct_indices = np.where(np.diff(yp_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, yt_sorted.size - 1]

    tps = np.cumsum(yt_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[yp_sorted[0] + 1.0, yp_sorted[threshold_idxs]]

    fpr = fps / max(fps[-1], 1)
    tpr = tps / max(tps[-1], 1)
    return fpr, tpr, thresholds


def roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


def precision_recall_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    desc_idx = np.argsort(yp)[::-1]
    yt_sorted = yt[desc_idx]
    yp_sorted = yp[desc_idx]

    distinct_indices = np.where(np.diff(yp_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, yt_sorted.size - 1]

    tps = np.cumsum(yt_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(np.sum(yt), 1)

    sl = slice(None, None, -1)
    return np.r_[precision[sl], 1.0], np.r_[recall[sl], 0.0], yp_sorted[threshold_idxs][sl]


def average_precision_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return float(-np.sum(np.diff(rec) * prec[:-1]))


def train_test_split_stratified(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y)
    train_indices, test_indices = [], []

    for cls_val in np.unique(y_arr):
        cls_idx = np.where(y_arr == cls_val)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(np.round(len(cls_idx) * test_size)))
        test_indices.extend(cls_idx[:n_test])
        train_indices.extend(cls_idx[n_test:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return (
        X.iloc[train_indices].copy(),
        X.iloc[test_indices].copy(),
        y_arr[train_indices],
        y_arr[test_indices],
    )


class StratifiedKFold:
    """Class-wise Stratified K-Fold cross-validator preserving class ratios per fold."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: Any, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        y_arr = np.asarray(y)
        folds = [[] for _ in range(self.n_splits)]

        for cls_val in np.unique(y_arr):
            cls_idx = np.where(y_arr == cls_val)[0]
            if self.shuffle:
                rng.shuffle(cls_idx)
            for fold_i, chunk in enumerate(np.array_split(cls_idx, self.n_splits)):
                folds[fold_i].extend(chunk.tolist())

        all_indices = np.arange(len(y_arr))
        for k in range(self.n_splits):
            val_idx = np.array(sorted(folds[k]), dtype=int)
            train_mask = np.ones(len(y_arr), dtype=bool)
            train_mask[val_idx] = False
            train_idx = all_indices[train_mask]
            yield train_idx, val_idx


def grid_search_cv_pipeline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    estimator_cls: Any,
    fixed_kwargs: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    cv: StratifiedKFold,
) -> Tuple[SoftVotingPipeline, Dict[str, Any], Dict[str, List[float]]]:
    """
    Executes Stratified K-Fold GridSearchCV over param_grid, selecting the hyperparameter
    combination maximizing out-of-fold ROC-AUC and returning CV fold metrics.
    """
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))

    fold_data = []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        pre = TabularPreprocessor()
        X_tr_mat = pre.fit_transform(X_train.iloc[tr_idx])
        X_val_mat = pre.transform(X_train.iloc[val_idx])
        fold_data.append((X_tr_mat, y_train[tr_idx], X_val_mat, y_train[val_idx]))

    best_auc = -1.0
    best_params = {}
    best_fold_metrics = {}

    for combo in combinations:
        cur_params = dict(zip(keys, combo))
        clean_kwargs = {k.replace("clf__", ""): v for k, v in cur_params.items()}
        merged_kwargs = {**fixed_kwargs, **clean_kwargs}

        aucs, f1s = [], []
        for X_tr_mat, y_tr_f, X_val_mat, y_val_f in fold_data:
            model = estimator_cls(**merged_kwargs)
            model.fit(X_tr_mat, y_tr_f)
            probs = model.predict_proba(X_val_mat)[:, 1]
            preds = (probs >= 0.50).astype(int)
            aucs.append(roc_auc_score(y_val_f, probs))
            f1s.append(f1_score(y_val_f, preds))

        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = cur_params
            best_fold_metrics = {"test_roc_auc": aucs, "test_f1": f1s}

    final_kwargs = {
        **fixed_kwargs,
        **{k.replace("clf__", ""): v for k, v in best_params.items()},
    }
    final_pre = TabularPreprocessor()
    final_clf = estimator_cls(**final_kwargs)
    best_pipe = SoftVotingPipeline(final_pre, final_clf).fit(X_train, y_train)
    return best_pipe, best_params, best_fold_metrics


def compute_permutation_importance(
    pipeline: SoftVotingPipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Computes model-agnostic Permutation Feature Importance on held-out test set."""
    rng = np.random.default_rng(random_state)
    base_probs = pipeline.predict_proba(X_test)[:, 1]
    base_auc = roc_auc_score(y_test, base_probs)

    records = []
    for col in X_test.columns:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            p_perm = pipeline.predict_proba(X_perm)[:, 1]
            drops.append(base_auc - roc_auc_score(y_test, p_perm))
        records.append(
            {
                "Feature": col,
                "Importance_Mean": float(np.mean(drops)),
                "Importance_Std": float(np.std(drops)),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(by="Importance_Mean", ascending=False)
        .reset_index(drop=True)
    )

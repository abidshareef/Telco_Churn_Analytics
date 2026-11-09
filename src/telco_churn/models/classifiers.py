"""
Vectorized Machine Learning Estimators for Telecom Churn Classification:
  - LogisticRegressionClassifier (L-BFGS-B with L2 regularization & balanced class weights)
  - DecisionTreeCART (Vectorized CART with weighted Gini impurity)
  - RandomForestEnsemble (Bagging & Extra-Trees randomized subspace ensembles)
  - GradientBoostingClassifierGBM (Newton-step log-odds stochastic gradient boosting)
  - MLPNeuralNetClassifier (2-Layer ReLU Feedforward Neural Network with Mini-Batch Adam)
  - WeightedSoftVotingEnsemble & SoftVotingPipeline
"""

import copy
from typing import List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from telco_churn.preprocessing.transformer import TabularPreprocessor
from telco_churn.preprocessing.features import engineer_domain_features
from telco_churn.utils.exceptions import ModelNotFittedError


class LogisticRegressionClassifier:
    """L2-Regularized Balanced Logistic Regression trained via L-BFGS-B optimization."""

    def __init__(self, C: float = 1.0, class_weight: str = "balanced", random_state: int = 42):
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        n_samples, n_features = X.shape
        y_arr = np.asarray(y, dtype=float)

        if self.class_weight == "balanced":
            n_pos = max(np.sum(y_arr == 1), 1.0)
            n_neg = max(np.sum(y_arr == 0), 1.0)
            w_pos = n_samples / (2.0 * n_pos)
            w_neg = n_samples / (2.0 * n_neg)
            sample_weights = np.where(y_arr == 1, w_pos, w_neg)
        else:
            sample_weights = np.ones(n_samples, dtype=float)

        reg_lambda = 1.0 / max(self.C * n_samples, 1e-6)

        def loss_and_grad(params):
            w = params[:-1]
            b = params[-1]
            z = X @ w + b
            p = expit(z)
            eps = 1e-12
            bce = -np.mean(
                sample_weights * (y_arr * np.log(p + eps) + (1.0 - y_arr) * np.log(1.0 - p + eps))
            )
            l2_pen = 0.5 * reg_lambda * np.sum(w ** 2)
            diff = (sample_weights * (p - y_arr)) / n_samples
            grad_w = X.T @ diff + reg_lambda * w
            grad_b = float(np.sum(diff))
            return bce + l2_pen, np.append(grad_w, grad_b)

        init_params = np.zeros(n_features + 1, dtype=float)
        res = minimize(
            loss_and_grad, init_params, jac=True, method="L-BFGS-B", options={"maxiter": 300}
        )
        self.coef_ = res.x[:-1]
        self.intercept_ = float(res.x[-1])
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.coef_ is None:
            raise ModelNotFittedError("LogisticRegressionClassifier is not fitted.")
        p1 = expit(X @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - p1, p1])


class FastCARTNode:
    __slots__ = ("feature", "threshold", "left", "right", "prob", "is_leaf")

    def __init__(self, prob: float):
        self.feature: int = -1
        self.threshold: float = 0.0
        self.left: Optional["FastCARTNode"] = None
        self.right: Optional["FastCARTNode"] = None
        self.prob: float = prob
        self.is_leaf: bool = True


class DecisionTreeCART:
    """Vectorized CART Decision Tree with class-balanced weighted Gini impurity."""

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        max_features: Optional[str] = None,
        random_splits: bool = False,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_splits = random_splits
        self.class_weight = class_weight
        self.random_state = random_state
        self.root: Optional[FastCARTNode] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeCART":
        self.rng_ = np.random.default_rng(self.random_state)
        y_arr = np.asarray(y, dtype=float)
        if self.class_weight in ("balanced", "balanced_subsample"):
            n_pos = max(np.sum(y_arr == 1), 1.0)
            n_neg = max(np.sum(y_arr == 0), 1.0)
            w = np.where(y_arr == 1, len(y_arr) / (2.0 * n_pos), len(y_arr) / (2.0 * n_neg))
        else:
            w = np.ones(len(y_arr), dtype=float)

        self.root = self._build_tree(X, y_arr, w, depth=0)
        self.is_fitted_ = True
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int) -> FastCARTNode:
        w_sum = float(np.sum(w))
        w_pos = float(np.sum(w * y))
        p_node = w_pos / max(w_sum, 1e-9)
        node = FastCARTNode(prob=p_node)

        n_samples, n_features = X.shape
        if (
            depth >= self.max_depth
            or n_samples < 2 * self.min_samples_leaf
            or p_node <= 1e-4
            or p_node >= 1.0 - 1e-4
        ):
            return node

        if self.max_features == "sqrt":
            k = max(int(np.sqrt(n_features)), 2)
            feat_indices = self.rng_.choice(n_features, size=min(k, n_features), replace=False)
        else:
            feat_indices = np.arange(n_features)

        parent_gini = 2.0 * p_node * (1.0 - p_node)
        best_gain = 1e-6
        best_feat = -1
        best_thresh = 0.0

        for f_idx in feat_indices:
            col = X[:, f_idx]
            c_min, c_max = float(np.min(col)), float(np.max(col))
            if c_max - c_min < 1e-9:
                continue

            if self.random_splits:
                thresholds = self.rng_.uniform(c_min, c_max, size=5)
            else:
                thresholds = np.unique(np.quantile(col, [0.15, 0.30, 0.50, 0.70, 0.85]))

            for thresh in thresholds:
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                w_left = float(np.sum(w[left_mask]))
                w_right = w_sum - w_left
                if w_left <= 1e-6 or w_right <= 1e-6:
                    continue

                p_left = float(np.sum(w[left_mask] * y[left_mask])) / w_left
                p_right = (w_pos - float(np.sum(w[left_mask] * y[left_mask]))) / w_right

                gini_left = 2.0 * p_left * (1.0 - p_left)
                gini_right = 2.0 * p_right * (1.0 - p_right)
                weighted_gini = (w_left / w_sum) * gini_left + (w_right / w_sum) * gini_right
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f_idx)
                    best_thresh = float(thresh)

        if best_feat == -1:
            return node

        left_mask = X[:, best_feat] <= best_thresh
        node.is_leaf = False
        node.feature = best_feat
        node.threshold = best_thresh
        node.left = self._build_tree(X[left_mask], y[left_mask], w[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], w[~left_mask], depth + 1)
        return node

    def _predict_node(self, node: FastCARTNode, X: np.ndarray, indices: np.ndarray, out: np.ndarray):
        if node.is_leaf or len(indices) == 0:
            out[indices] = node.prob
            return
        left_mask = X[indices, node.feature] <= node.threshold
        self._predict_node(node.left, X, indices[left_mask], out)
        self._predict_node(node.right, X, indices[~left_mask], out)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.root is None:
            raise ModelNotFittedError("DecisionTreeCART is not fitted.")
        out = np.zeros(len(X), dtype=float)
        self._predict_node(self.root, X, np.arange(len(X)), out)
        return np.column_stack([1.0 - out, out])


class RandomForestEnsemble:
    """Bagging or Extra-Trees Ensemble of feature-subsampled DecisionTreeCART estimators."""

    def __init__(
        self,
        n_estimators: int = 45,
        max_depth: int = 8,
        min_samples_leaf: int = 15,
        random_splits: bool = False,
        bootstrap: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_splits = random_splits
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees_: List[DecisionTreeCART] = []
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestEnsemble":
        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)
        self.trees_ = []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                boot_idx = rng.integers(0, n_samples, size=n_samples)
                X_b, y_b = X[boot_idx], y[boot_idx]
            else:
                X_b, y_b = X, y

            tree = DecisionTreeCART(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt",
                random_splits=self.random_splits,
                class_weight="balanced_subsample",
                random_state=int(rng.integers(1, 1_000_000)),
            )
            tree.fit(X_b, y_b)
            self.trees_.append(tree)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or not self.trees_:
            raise ModelNotFittedError("RandomForestEnsemble is not fitted.")
        preds = np.mean([t.predict_proba(X)[:, 1] for t in self.trees_], axis=0)
        return np.column_stack([1.0 - preds, preds])


class RegressionStumpTree:
    """Shallow regression tree fitted on pseudo-residuals with Newton-Raphson leaf updates."""

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root: Optional[FastCARTNode] = None

    def fit(self, X: np.ndarray, residuals: np.ndarray, hessians: np.ndarray, rng: np.random.Generator):
        self.root = self._build(X, residuals, hessians, depth=0, rng=rng)
        return self

    def _build(
        self, X: np.ndarray, r: np.ndarray, h: np.ndarray, depth: int, rng: np.random.Generator
    ) -> FastCARTNode:
        leaf_val = float(np.sum(r) / (np.sum(h) + 1.0))
        node = FastCARTNode(prob=leaf_val)
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < 2 * self.min_samples_leaf:
            return node

        total_sum = float(np.sum(r))
        base_score = (total_sum ** 2) / n_samples
        best_gain = 1e-6
        best_feat = -1
        best_thresh = 0.0

        feat_sub = rng.choice(n_features, size=max(int(n_features * 0.75), 3), replace=False)
        for f_idx in feat_sub:
            col = X[:, f_idx]
            thresholds = np.unique(np.quantile(col, [0.20, 0.40, 0.60, 0.80]))
            for thresh in thresholds:
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                s_left = float(np.sum(r[left_mask]))
                s_right = total_sum - s_left
                gain = (s_left ** 2) / n_left + (s_right ** 2) / n_right - base_score
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f_idx)
                    best_thresh = float(thresh)

        if best_feat == -1:
            return node

        left_mask = X[:, best_feat] <= best_thresh
        node.is_leaf = False
        node.feature = best_feat
        node.threshold = best_thresh
        node.left = self._build(X[left_mask], r[left_mask], h[left_mask], depth + 1, rng)
        node.right = self._build(X[~left_mask], r[~left_mask], h[~left_mask], depth + 1, rng)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(X), dtype=float)
        self._pred_rec(self.root, X, np.arange(len(X)), out)
        return out

    def _pred_rec(self, node: FastCARTNode, X: np.ndarray, indices: np.ndarray, out: np.ndarray):
        if node.is_leaf or len(indices) == 0:
            out[indices] = node.prob
            return
        left_mask = X[indices, node.feature] <= node.threshold
        self._pred_rec(node.left, X, indices[left_mask], out)
        self._pred_rec(node.right, X, indices[~left_mask], out)


class GradientBoostingClassifierGBM:
    """Stochastic Gradient Boosted Decision Trees optimizing Binomial Deviance."""

    def __init__(
        self,
        n_estimators: int = 75,
        learning_rate: float = 0.08,
        max_depth: int = 3,
        subsample: float = 0.85,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.init_logit_: float = 0.0
        self.trees_: List[RegressionStumpTree] = []
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifierGBM":
        rng = np.random.default_rng(self.random_state)
        y_arr = np.asarray(y, dtype=float)
        p_mean = np.clip(np.mean(y_arr), 1e-4, 1.0 - 1e-4)
        self.init_logit_ = float(np.log(p_mean / (1.0 - p_mean)))

        f_pred = np.full(len(y_arr), self.init_logit_, dtype=float)
        self.trees_ = []
        n_sub = max(int(len(y_arr) * self.subsample), 2)

        for _ in range(self.n_estimators):
            p = expit(f_pred)
            residuals = y_arr - p
            hessians = np.maximum(p * (1.0 - p), 1e-6)

            sub_idx = rng.choice(len(y_arr), size=n_sub, replace=False)
            tree = RegressionStumpTree(max_depth=self.max_depth, min_samples_leaf=20)
            tree.fit(X[sub_idx], residuals[sub_idx], hessians[sub_idx], rng)
            f_pred += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or not self.trees_:
            raise ModelNotFittedError("GradientBoostingClassifierGBM is not fitted.")
        f_pred = np.full(len(X), self.init_logit_, dtype=float)
        for tree in self.trees_:
            f_pred += self.learning_rate * tree.predict(X)
        p1 = expit(f_pred)
        return np.column_stack([1.0 - p1, p1])


class MLPNeuralNetClassifier:
    """Two-Layer Feedforward Neural Network (ReLU + Sigmoid) trained via Mini-Batch Adam."""

    def __init__(
        self,
        hidden_dim1: int = 48,
        hidden_dim2: int = 24,
        alpha: float = 0.02,
        learning_rate_init: float = 0.005,
        epochs: int = 40,
        random_state: int = 42,
    ):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.epochs = epochs
        self.random_state = random_state
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPNeuralNetClassifier":
        rng = np.random.default_rng(self.random_state)
        n_samples, n_in = X.shape
        y_col = np.asarray(y, dtype=float).reshape(-1, 1)

        pos_w = n_samples / (2.0 * max(np.sum(y_col == 1), 1.0))
        neg_w = n_samples / (2.0 * max(np.sum(y_col == 0), 1.0))

        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, self.hidden_dim1))
        self.b1 = np.zeros((1, self.hidden_dim1))
        self.W2 = rng.normal(
            0, np.sqrt(2.0 / self.hidden_dim1), size=(self.hidden_dim1, self.hidden_dim2)
        )
        self.b2 = np.zeros((1, self.hidden_dim2))
        self.W3 = rng.normal(0, np.sqrt(1.0 / self.hidden_dim2), size=(self.hidden_dim2, 1))
        self.b3 = np.zeros((1, 1))

        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        batch_size = min(256, max(n_samples, 1))

        for _ in range(self.epochs):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                step += 1
                idx = perm[start : start + batch_size]
                xb, yb = X[idx], y_col[idx]
                wb = np.where(yb == 1, pos_w, neg_w)
                bs = len(xb)

                z1 = xb @ self.W1 + self.b1
                a1 = np.maximum(0.0, z1)
                z2 = a1 @ self.W2 + self.b2
                a2 = np.maximum(0.0, z2)
                z3 = a2 @ self.W3 + self.b3
                p = expit(z3)

                dz3 = (wb * (p - yb)) / bs
                dW3 = a2.T @ dz3 + (self.alpha / n_samples) * self.W3
                db3 = np.sum(dz3, axis=0, keepdims=True)

                da2 = dz3 @ self.W3.T
                dz2 = da2 * (z2 > 0)
                dW2 = a1.T @ dz2 + (self.alpha / n_samples) * self.W2
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0)
                dW1 = xb.T @ dz1 + (self.alpha / n_samples) * self.W1
                db1 = np.sum(dz1, axis=0, keepdims=True)

                grads = [dW1, db1, dW2, db2, dW3, db3]
                for i, (param, g) in enumerate(zip(params, grads)):
                    m[i] = beta1 * m[i] + (1.0 - beta1) * g
                    v[i] = beta2 * v[i] + (1.0 - beta2) * (g ** 2)
                    m_hat = m[i] / (1.0 - beta1 ** step)
                    v_hat = v[i] / (1.0 - beta2 ** step)
                    param -= self.learning_rate_init * m_hat / (np.sqrt(v_hat) + eps)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ModelNotFittedError("MLPNeuralNetClassifier is not fitted.")
        a1 = np.maximum(0.0, X @ self.W1 + self.b1)
        a2 = np.maximum(0.0, a1 @ self.W2 + self.b2)
        p1 = expit(a2 @ self.W3 + self.b3).ravel()
        return np.column_stack([1.0 - p1, p1])


class WeightedSoftVotingEnsemble:
    """Weighted Soft-Voting Meta-Ensemble combining calibrated probabilities."""

    def __init__(self, estimators: List[Tuple[str, Any]], weights: List[float]):
        self.estimators = estimators
        self.weights = np.array(weights, dtype=float) / np.sum(weights)
        self.fitted_estimators_: List[Any] = []
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedSoftVotingEnsemble":
        self.fitted_estimators_ = []
        for _, est in self.estimators:
            cloned = copy.deepcopy(est)
            cloned.fit(X, y)
            self.fitted_estimators_.append(cloned)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or not self.fitted_estimators_:
            raise ModelNotFittedError("WeightedSoftVotingEnsemble is not fitted.")
        prob_sum = np.zeros(len(X), dtype=float)
        for w, est in zip(self.weights, self.fitted_estimators_):
            prob_sum += w * est.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - prob_sum, prob_sum])


class SoftVotingPipeline:
    """End-to-end Pipeline combining TabularPreprocessor + Estimator."""

    def __init__(self, preprocessor: TabularPreprocessor, clf: Any, optimal_threshold: float = 0.50):
        self.preprocessor = preprocessor
        self.clf = clf
        self.optimal_threshold = optimal_threshold
        self.named_steps = {"preprocessor": preprocessor, "clf": clf}
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SoftVotingPipeline":
        Xt = self.preprocessor.fit_transform(X, y)
        self.clf.fit(Xt, y)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise ModelNotFittedError("SoftVotingPipeline is not fitted.")
        Xt = self.preprocessor.transform(X)
        return self.clf.predict_proba(Xt)

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)[:, 1]
        return (probs >= thresh).astype(int)

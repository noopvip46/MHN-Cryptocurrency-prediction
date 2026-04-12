# HopCPT: conformal prediction wrapper for any BaseFlashCrashModel.
#
# Conformal prediction gives distribution-free coverage guarantees. After the base model
# is trained, we run it on a held-out calibration set, compute nonconformity scores, and
# store a quantile q_hat. At test time, a class is included in the prediction set if its
# nonconformity score falls below q_hat. This guarantees P(true label in set) >= 1 - alpha.
#
# Nonconformity score: s = 1 - p(true class | X)   (higher = more surprising)
# Prediction set rule: include class c if 1 - p(c | X) <= q_hat
#
# The four possible outcomes for binary classification are:
#   1  = only crash in the set -> predict crash
#   0  = only no-crash in the set -> predict no crash
#   2  = both classes in the set -> uncertain
#  -1  = empty set -> abstain
#
# Reference: Angelopoulos & Bates "A Gentle Introduction to Conformal Prediction
#            and Distribution-Free Uncertainty Quantification." arXiv 2107.07511 (2021).

import math

import numpy as np

from models.base import BaseFlashCrashModel


class HopCPT:
    # This class does not inherit BaseFlashCrashModel because conformal prediction
    # needs a separate calibration step and returns set-valued predictions, not a single label.

    def __init__(self, base_model, alpha=0.10):
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.base_model = base_model
        self.alpha  = alpha
        self._q_hat = None  # set by calibrate()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Just delegates to the wrapped base model
        self.base_model.fit(X_train, y_train, X_val, y_val)
        return self

    def calibrate(self, X_cal, y_cal):
        # Compute nonconformity scores on the calibration set and store q_hat.
        # q_hat is the ceil((n+1)(1-alpha))/n quantile of the scores, which
        # gives the desired marginal coverage guarantee.
        y_cal = np.asarray(y_cal, dtype=int)
        probs = self.base_model.predict_proba(X_cal)
        n     = len(y_cal)

        # For y=1: s = 1 - p(crash), for y=0: s = p(crash)
        scores = np.where(y_cal == 1, 1.0 - probs, probs)

        level       = min(math.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self._q_hat = float(np.quantile(scores, level))

        n_pos = int(y_cal.sum())
        print(f"[HopCPT] calibrated on {n} samples (pos={n_pos})  alpha={self.alpha}  q_hat={self._q_hat:.4f}  target_coverage={1 - self.alpha:.2%}")
        return self

    def predict_set(self, X):
        # Returns set-valued predictions: 1=crash, 0=no crash, 2=uncertain, -1=empty (abstain)
        if self._q_hat is None:
            raise RuntimeError("Call calibrate() before predict_set().")
        probs = self.base_model.predict_proba(X)

        crash_in    = (1.0 - probs) <= self._q_hat
        no_crash_in = probs <= self._q_hat

        result = np.full(len(X), -1, dtype=np.int8)
        result[ crash_in & ~no_crash_in] = 1
        result[~crash_in &  no_crash_in] = 0
        result[ crash_in &  no_crash_in] = 2
        return result

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict(self, X, threshold=0.5):
        return self.base_model.predict(X, threshold=threshold)

    def evaluate(self, X, y):
        # Base model metrics plus conformal coverage statistics.
        y       = np.asarray(y, dtype=int)
        metrics = self.base_model.evaluate(X, y)

        if self._q_hat is not None:
            pred_sets = self.predict_set(X)
            uncertain = pred_sets == 2
            empty     = pred_sets == -1

            true_in_set = (
                ((y == 1) & ((pred_sets == 1) | uncertain)) |
                ((y == 0) & ((pred_sets == 0) | uncertain))
            )
            metrics["conformal_coverage"] = float(true_in_set.mean())
            metrics["uncertain_rate"]     = float(uncertain.mean())
            metrics["empty_set_rate"]     = float(empty.mean())
        else:
            metrics["conformal_coverage"] = float("nan")
            metrics["uncertain_rate"]     = float("nan")
            metrics["empty_set_rate"]     = float("nan")

        metrics["target_coverage"] = 1.0 - self.alpha
        return metrics

    def __repr__(self):
        return f"HopCPT(base={self.base_model.name!r}, alpha={self.alpha}, calibrated={self._q_hat is not None})"

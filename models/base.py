# Abstract base class that every spike-correction model in this project inherits from.
# Subclasses must implement fit and predict_proba.
# predict and evaluate are provided here and work automatically once those two are implemented.

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


class BaseFlashCrashModel(ABC):

    # Each subclass should set this to a short name string e.g. "lstm"
    name: str = "base"

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Train the model. Sequence models expect X shape (N, seq_len, n_features),
        # flat/sklearn models expect (N, n_features). Returns self.
        ...

    @abstractmethod
    def predict_proba(self, X):
        # Return predicted crash probabilities, shape (N,), values in [0, 1].
        ...

    def predict(self, X, threshold=0.5):
        # Hard binary predictions using a probability threshold.
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X, y):
        # Metrics for binary classification (reversal vs continuation).
        #
        # With CUSUM + Triple Barrier labeling, the class balance is typically
        # closer to 50/50 than the old crash labels (~1:50), but can still be
        # imbalanced depending on barrier parameters and market regime.
        #
        # Threshold-free (primary):
        #   roc_auc    — overall discrimination ability
        #   avg_prec   — area under precision-recall curve
        #
        # Threshold-dependent (all computed at the optimal F1 threshold):
        #   balanced_acc — mean of per-class recall; unaffected by class imbalance
        #   mcc          — Matthews Correlation Coefficient [-1, 1], 0 = random
        #   f1_macro     — mean of F1 for each class
        #   f1           — F1 for the positive (reversal) class
        #   precision / recall — at the same optimal threshold
        #
        # The optimal threshold is found by sweeping the precision-recall curve.
        y = np.asarray(y, dtype=int)
        probs = self.predict_proba(X)

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            print(f"[{self.name}] evaluate: only one class present "
                  f"(pos={n_pos}, neg={n_neg}), some metrics will be 0.")

        try:
            roc_auc = roc_auc_score(y, probs)
        except ValueError:
            roc_auc = float("nan")

        try:
            avg_prec = average_precision_score(y, probs)
        except ValueError:
            avg_prec = float("nan")

        # Sweep PR curve to find threshold that maximises positive-class F1.
        try:
            prec_arr, rec_arr, thr_arr = precision_recall_curve(y, probs)
            prec_t = prec_arr[:-1]
            rec_t  = rec_arr[:-1]
            f1_t   = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-9)
            best_i = int(f1_t.argmax())
            best_thr = float(thr_arr[best_i])
        except Exception:
            best_thr = 0.5

        preds = (probs >= best_thr).astype(int)

        return {
            # ── threshold-free ────────────────────────────────────────────────
            "roc_auc":       roc_auc,
            "avg_prec":      avg_prec,
            # ── threshold-dependent (at optimal F1 threshold) ─────────────────
            "opt_threshold": best_thr,
            "balanced_acc":  balanced_accuracy_score(y, preds),
            "mcc":           matthews_corrcoef(y, preds),
            "f1_macro":      f1_score(y, preds, average="macro",    zero_division=0),
            "f1":            f1_score(y, preds, average="binary",   zero_division=0),
            "precision":     precision_score(y, preds,              zero_division=0),
            "recall":        recall_score(y, preds,                 zero_division=0),
        }

    def save(self, path):
        """Save model weights to path.  Implemented by each DL subclass."""
        raise NotImplementedError(f"{self.name}.save() is not implemented.")

    def load(self, path):
        """Load model weights from path.  Implemented by each DL subclass."""
        raise NotImplementedError(f"{self.name}.load() is not implemented.")

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"

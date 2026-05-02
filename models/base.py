# Abstract base class that every flash crash model in this project inherits from.
# Subclasses must implement fit and predict_proba.
# predict and evaluate are provided here and work automatically once those two are implemented.

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
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
        # Compute a standard set of binary classification metrics.
        # We use average precision (area under PR curve) as the primary metric
        # since flash crash datasets are heavily imbalanced — accuracy is meaningless here.
        y = np.asarray(y, dtype=int)
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            print(f"[{self.name}] evaluate: only one class present in y (pos={n_pos}, neg={n_neg}), some metrics will be 0.")

        try:
            roc_auc = roc_auc_score(y, probs)
        except ValueError:
            roc_auc = float("nan")

        try:
            avg_prec = average_precision_score(y, probs)
        except ValueError:
            avg_prec = float("nan")

        return {
            "roc_auc":   roc_auc,
            "avg_prec":  avg_prec,
            "f1":        f1_score(y, preds, zero_division=0),
            "precision": precision_score(y, preds, zero_division=0),
            "recall":    recall_score(y, preds, zero_division=0),
        }

    def save(self, path):
        """Save model weights to path.  Implemented by each DL subclass."""
        raise NotImplementedError(f"{self.name}.save() is not implemented.")

    def load(self, path):
        """Load model weights from path.  Implemented by each DL subclass."""
        raise NotImplementedError(f"{self.name}.load() is not implemented.")

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"

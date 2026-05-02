# ML baselines: XGBoost, Random Forest, and Logistic Regression wrapped in the
# BaseFlashCrashModel interface so they slot into the same pipeline as the deep models.
# All three handle class imbalance internally.
# Input must be flattened — use SequenceDataset.get_flat_splits() to get the right format.
#
# GPU support: only XGBoost benefits from device="cuda".  Random Forest and Logistic
# Regression are sklearn models and always run on CPU regardless of the device flag.

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from models.base import BaseFlashCrashModel

# XGBoost is optional so we can import the module even without it installed;
# the error is only raised when you actually try to construct an xgboost model.
try:
    import xgboost as _xgb
    from xgboost import XGBClassifier as _XGBClassifier
    _XGBOOST_AVAILABLE = True
    # XGBoost 2.0 unified the GPU API: device="cuda" replaces tree_method="gpu_hist".
    _XGB_VERSION = tuple(int(x) for x in _xgb.__version__.split(".")[:2])
except ImportError:
    _XGBOOST_AVAILABLE = False
    _XGB_VERSION = (0, 0)

_VALID_TYPES = ("xgboost", "random_forest", "logistic")


class MLBaselinesModel(BaseFlashCrashModel):

    def __init__(self, model_type="xgboost", device="cpu", **kwargs):
        if model_type not in _VALID_TYPES:
            raise ValueError(f"model_type must be one of {_VALID_TYPES}, got {model_type!r}")
        self.model_type   = model_type
        self.device       = device   # only used by XGBoost; RF and LR always run on CPU
        self.kwargs       = kwargs
        self.name         = f"baseline_{model_type}"
        self._model       = None
        self._feature_dim = None

    def _build_model(self, pos_weight):
        # Construct the underlying estimator with class imbalance handling baked in.
        if self.model_type == "xgboost":
            if not _XGBOOST_AVAILABLE:
                raise ImportError("xgboost is not installed. Run: pip install xgboost")
            defaults = dict(
                scale_pos_weight=pos_weight,
                tree_method="hist",
                eval_metric="aucpr",
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=0,
                random_state=42,
            )
            # GPU acceleration — XGBoost 2.0+ uses device="cuda"; older uses tree_method="gpu_hist"
            if self.device == "cuda":
                if _XGB_VERSION >= (2, 0):
                    defaults["device"] = "cuda"
                else:
                    defaults["tree_method"] = "gpu_hist"
                    defaults["gpu_id"]      = 0
                print(f"  [{self.name}] using XGBoost GPU  (version {'.'.join(str(x) for x in _XGB_VERSION)})")
            else:
                print(f"  [{self.name}] using XGBoost CPU  (pass --device cuda to use GPU)")

            defaults.update(self.kwargs)
            try:
                return _XGBClassifier(**defaults)
            except TypeError:
                # Older XGBoost versions may not accept some newer params
                defaults.pop("use_label_encoder", None)
                return _XGBClassifier(**defaults)

        elif self.model_type == "random_forest":
            defaults = dict(n_estimators=300, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=42)
            defaults.update(self.kwargs)
            return RandomForestClassifier(**defaults)

        else:  # logistic
            defaults = dict(class_weight="balanced", max_iter=1000, solver="lbfgs", C=1.0, random_state=42, n_jobs=-1)
            defaults.update(self.kwargs)
            return LogisticRegression(**defaults)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # X_val and y_val are only used by XGBoost for eval_set; ignored by the others.
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=int)
        self._feature_dim = X_train.shape[1]

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_weight = n_neg / max(n_pos, 1)
        print(f"[{self.name}] training  X={X_train.shape}  pos={n_pos}  neg={n_neg}  pos_weight={pos_weight:.2f}")

        self._model = self._build_model(pos_weight)

        if self.model_type == "xgboost" and X_val is not None and y_val is not None:
            self._model.fit(
                X_train, y_train,
                eval_set=[(np.asarray(X_val, dtype=np.float32), np.asarray(y_val, dtype=int))],
                verbose=10,   # print aucpr on val set every 10 trees
            )
        elif self.model_type == "xgboost":
            self._model.fit(X_train, y_train, verbose=10)
        else:
            self._model.fit(X_train, y_train)

        print(f"[{self.name}] training complete.")
        return self

    def predict_proba(self, X):
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        # sklearn and XGBoost both return (N, 2); column 1 is P(crash)
        return self._model.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]

    def feature_importance(self):
        # Returns a sorted dict of {feature_index: importance_score} for tree models.
        # For logistic regression returns absolute coefficient values as a proxy.
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if self.model_type in ("xgboost", "random_forest"):
            scores = {f"feature_{i}": float(v) for i, v in enumerate(self._model.feature_importances_)}
        elif self.model_type == "logistic":
            scores = {f"feature_{i}": float(v) for i, v in enumerate(np.abs(self._model.coef_[0]))}
        else:
            return {}

        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

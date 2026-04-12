"""
models — modular ML adapter layer for flash-crash prediction.

Public API
----------
SequenceDataset          : sliding-window data loader (data_adapter.py)

BaseFlashCrashModel      : abstract base class (base.py)

HopCPT                   : conformal prediction wrapper (hopfield/hopcpt.py)
MHNFlashCrashModel       : Modern Hopfield Network (hopfield/mhn.py)
STanHopModel             : Sparse Tandem Hopfield Net (hopfield/stanhop.py)

LSTMFlashCrashModel      : Bidirectional LSTM + attention pooling
TransformerFlashCrashModel: Transformer encoder with CLS token

MLBaselinesModel         : XGBoost / Random Forest / Logistic Regression
"""

from .base import BaseFlashCrashModel
from .baselines import MLBaselinesModel
from .data_adapter import SequenceDataset
from .deep_learning import LSTMFlashCrashModel, TransformerFlashCrashModel
from .hopfield import HopCPT, MHNFlashCrashModel, STanHopModel

__all__ = [
    "SequenceDataset",
    "BaseFlashCrashModel",
    "HopCPT",
    "MHNFlashCrashModel",
    "STanHopModel",
    "LSTMFlashCrashModel",
    "TransformerFlashCrashModel",
    "MLBaselinesModel",
]

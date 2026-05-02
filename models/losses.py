# Shared loss functions for all DL flash-crash models.

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for heavily imbalanced datasets.

    Down-weights easy negatives (the overwhelming majority class) so gradient
    updates focus on the hard, ambiguous examples near the decision boundary.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * BCE(logit, target)

    Parameters
    ----------
    alpha : float
        Class-balance weight for the *positive* class, in [0, 1].
        Should be set to  n_neg / (n_pos + n_neg)  so rare positives get
        proportionally more weight.  For a 50:1 imbalance this is ~0.98.
    gamma : float
        Focusing exponent (default 2.0 from the original paper).
        gamma=0 reduces to standard weighted BCE.
        Higher values push the model harder to correctly classify
        already-difficult examples.
    """

    def __init__(self, alpha: float = 0.99, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Element-wise BCE (no reduction yet)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t: probability assigned to the *correct* class
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1.0 - probs) * (1.0 - targets)

        # alpha_t: per-sample class weight
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
        return loss.mean()

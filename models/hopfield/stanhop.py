# STanHop-Net: Sparse Tandem Hopfield Network for multivariate time series.
# Reference: Wu et al. "STanHop: Sparse Tandem Hopfield Model for Memory-Enhanced
#            Time Series Prediction." arXiv 2312.00340 (2023).
#
# Two tandem sparse Hopfield layers run in sequence:
#   1) Temporal layer — attends across time steps
#   2) Variate layer  — attends across feature channels
# Sparsity is enforced by keeping only the top-k attention scores per query (the rest become -inf).
# Both branches are pooled and concatenated before the classification head.

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import BaseFlashCrashModel
from models.data_adapter import WindowDataset
from models.losses import FocalLoss


class LearnablePositionalEncoding(nn.Module):
    # Learnable additive positional encoding, one vector per time step.

    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe


class SparseHopfieldLayer(nn.Module):
    # Single sparse Hopfield layer with top-k masking.
    # Only the top-k attention scores are kept per query; the rest are set to -inf before softmax.

    def __init__(self, d_model, n_heads, top_k, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.beta    = 1.0 / math.sqrt(self.d_head)
        self.top_k   = top_k

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq, d_model) where seq is either the time or variate dimension
        B, S, _ = x.shape
        H, D = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
        K = self.k_proj(x).view(B, S, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, S, H, D).transpose(1, 2)

        scores = self.beta * torch.matmul(Q, K.transpose(-2, -1))  # (B, H, S, S)

        # Keep only the top-k scores, mask the rest to -inf so they vanish after softmax
        k = min(self.top_k, S)
        if k < S:
            threshold, _ = scores.topk(k, dim=-1)
            scores = scores.masked_fill(scores < threshold[..., -1:].expand_as(scores), float("-inf"))

        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, H * D)
        return self.norm(self.out_proj(out) + x)


class _STanHopNet(nn.Module):
    # Full STanHop network used internally by STanHopModel.

    def __init__(self, seq_len, n_features, hidden_dim, n_heads, top_k, dropout):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pos_enc          = LearnablePositionalEncoding(seq_len, hidden_dim)
        self.temporal_hopfield = SparseHopfieldLayer(hidden_dim, n_heads, top_k, dropout)
        self.variate_proj     = nn.Linear(seq_len, seq_len)   # mixes time steps before the variate layer
        self.variate_hopfield  = SparseHopfieldLayer(hidden_dim, n_heads, top_k, dropout)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, T, n_features) -> logits: (B,)
        h = self.pos_enc(self.input_proj(x))   # (B, T, hidden_dim)

        # 1) Temporal Hopfield — attends across the time dimension
        h_temp = self.temporal_hopfield(h)     # (B, T, hidden_dim)

        # 2) Variate Hopfield — attends across the feature (variate) dimension.
        #    We transpose so the layer sees (B, hidden_dim, T) and apply a linear
        #    to mix time steps, then transpose back before the Hopfield layer.
        h_var = self.variate_hopfield(self.variate_proj(h_temp.transpose(1, 2)).transpose(1, 2))

        # Pool both branches over time and concatenate before the head
        combined = torch.cat([h_temp.mean(dim=1), h_var.mean(dim=1)], dim=-1)  # (B, 2*hidden_dim)
        return self.classifier(combined).squeeze(-1)


class STanHopModel(BaseFlashCrashModel):

    name = "stanhop"

    def __init__(self, seq_len, n_features, hidden_dim=128, n_heads=4, top_k=10,
                 dropout=0.1, lr=1e-3, epochs=50, batch_size=256, device="auto"):
        self.seq_len    = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.top_k      = top_k
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self._net = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            checkpoint_dir=None, resume_from=None):
        print(f"[{self.name}] training on {self.device}  X={X_train.shape}  pos_frac={y_train.mean():.4f}")

        self._net = _STanHopNet(
            seq_len=self.seq_len, n_features=self.n_features, hidden_dim=self.hidden_dim,
            n_heads=self.n_heads, top_k=self.top_k, dropout=self.dropout,
        ).to(self.device)

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        alpha    = n_neg / max(n_pos + n_neg, 1)   # ≈ 0.98 for 50:1 imbalance
        criterion = FocalLoss(alpha=alpha, gamma=2.0).to(self.device)

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        start_epoch = 1
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self._net.load_state_dict(ckpt["state_dict"])
            if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"  [{self.name}] resumed from epoch {ckpt.get('epoch', '?')} → continuing from {start_epoch}")

        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        def _save_checkpoint(epoch):
            p = (ckpt_dir or Path("checkpoints")) / f"{self.name}_checkpoint.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_name": self.name,
                "epoch":      epoch,
                "state_dict": self._net.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
            }, p)
            return p

        loader = DataLoader(
            WindowDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=True, drop_last=False,
            num_workers=0,
        )

        self._last_epoch = start_epoch - 1
        try:
            self._net.train()
            for epoch in range(start_epoch, self.epochs + 1):
                epoch_loss = 0.0
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(self._net(X_batch), y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item() * len(X_batch)

                scheduler.step()
                epoch_loss /= len(X_train)
                self._last_epoch = epoch

                if ckpt_dir:
                    _save_checkpoint(epoch)

                # Print loss every epoch; val metrics every 5 epochs (evaluate is expensive)
                msg = f"[{self.name}] epoch {epoch:>3}/{self.epochs}  loss={epoch_loss:.5f}"
                if X_val is not None and y_val is not None and (epoch % 5 == 0 or epoch == self.epochs):
                    val = self.evaluate(X_val, y_val)
                    msg += f"  val_auc={val['roc_auc']:.4f}  val_ap={val['avg_prec']:.4f}"
                print(msg, flush=True)

        except KeyboardInterrupt:
            p = _save_checkpoint(self._last_epoch)
            print(f"\n  [{self.name}] interrupted at epoch {self._last_epoch} — checkpoint saved → {p}")
            raise

        return self

    def save(self, path):
        """Save network weights (no optimizer state) for inference / export."""
        if self._net is None:
            raise RuntimeError("No trained network. Call fit() first.")
        torch.save({
            "model_name": self.name,
            "epoch":      getattr(self, "_last_epoch", self.epochs),
            "state_dict": self._net.state_dict(),
        }, path)
        print(f"  [{self.name}] saved → {path}")

    def load(self, path):
        """Load network weights into an already-initialised model (same hparams required)."""
        if self._net is None:
            raise RuntimeError("Initialise the network first by calling fit() once (even with dummy data), "
                               "or use --resume in run.py which handles reconstruction automatically.")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._net.load_state_dict(ckpt["state_dict"])
        self._last_epoch = ckpt.get("epoch", 0)
        print(f"  [{self.name}] loaded from {path}  (epoch {self._last_epoch})")

    def predict_proba(self, X):
        if self._net is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        was_training = self._net.training
        self._net.eval()
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.from_numpy(
                    np.ascontiguousarray(X[start : start + self.batch_size])
                ).to(self.device)
                all_probs.append(torch.sigmoid(self._net(batch)).cpu().numpy())
        if was_training:
            self._net.train()
        return np.concatenate(all_probs, axis=0)

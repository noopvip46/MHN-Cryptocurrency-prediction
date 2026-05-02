# Modern Hopfield Network (MHN) for flash crash prediction.
# Reference: Ramsauer et al. "Hopfield Networks is All You Need." ICLR 2021.
#
# The architecture is: input projection -> ModernHopfieldLayer (with learned memory patterns)
# -> average pool over time -> classification head.
# The memory patterns are learnable parameters so the network learns to store
# and retrieve crash precursor patterns end-to-end.

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import BaseFlashCrashModel
from models.data_adapter import WindowDataset


class ModernHopfieldLayer(nn.Module):
    # Multi-head softmax attention where keys and values come from learned memory patterns
    # rather than the input itself. beta = 1/sqrt(d_head) is the Hopfield scaling factor.

    def __init__(self, input_dim, hidden_dim, n_heads, n_patterns, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head  = hidden_dim // n_heads
        self.beta    = 1.0 / math.sqrt(self.d_head)

        self.query_proj = nn.Linear(input_dim, hidden_dim)

        # Learned memory patterns — the network stores crash precursor patterns here
        self.memory_keys   = nn.Parameter(torch.randn(n_patterns, hidden_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(n_patterns, hidden_dim) * 0.02)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> out: (batch, seq_len, hidden_dim)
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        Q = self.query_proj(x).view(B, T, H, D).transpose(1, 2)        # (B, H, T, D)
        K = self.memory_keys.view(-1, H, D).transpose(0, 1).unsqueeze(0).expand(B, -1, -1, -1)    # (B, H, n_patterns, D)
        V = self.memory_values.view(-1, H, D).transpose(0, 1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, n_patterns, D)

        scores = self.beta * torch.matmul(Q, K.transpose(-2, -1))  # (B, H, T, n_patterns)
        attn   = self.dropout(torch.softmax(scores, dim=-1))
        out    = torch.matmul(attn, V)                              # (B, H, T, D)
        out    = out.transpose(1, 2).contiguous().view(B, T, H * D)
        out    = self.norm(self.out_proj(out) + Q.transpose(1, 2).contiguous().view(B, T, H * D))
        return out


class _MHNNet(nn.Module):
    # Full MHN network used internally by MHNFlashCrashModel

    def __init__(self, seq_len, n_features, hidden_dim, n_heads, n_memory_patterns, dropout):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.hopfield = ModernHopfieldLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim,
            n_heads=n_heads, n_patterns=n_memory_patterns, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: (B, seq_len, n_features) -> logits: (B,)
        h = self.input_proj(x)   # (B, T, hidden_dim)
        h = self.hopfield(h)     # (B, T, hidden_dim)
        h = h.mean(dim=1)        # average pool over time -> (B, hidden_dim)
        return self.classifier(h).squeeze(-1)


class MHNFlashCrashModel(BaseFlashCrashModel):

    name = "mhn"

    def __init__(self, seq_len, n_features, hidden_dim=128, n_heads=4,
                 n_memory_patterns=64, dropout=0.1, lr=1e-3, epochs=50,
                 batch_size=256, device="auto"):
        self.seq_len           = seq_len
        self.n_features        = n_features
        self.hidden_dim        = hidden_dim
        self.n_heads           = n_heads
        self.n_memory_patterns = n_memory_patterns
        self.dropout           = dropout
        self.lr                = lr
        self.epochs            = epochs
        self.batch_size        = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self._net = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            checkpoint_dir=None, resume_from=None):
        print(f"[{self.name}] training on {self.device}  X={X_train.shape}  pos_frac={y_train.mean():.4f}")

        self._net = _MHNNet(
            seq_len=self.seq_len, n_features=self.n_features, hidden_dim=self.hidden_dim,
            n_heads=self.n_heads, n_memory_patterns=self.n_memory_patterns, dropout=self.dropout,
        ).to(self.device)

        # Weight the positive class to handle the heavy class imbalance
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
        self._net.eval()
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.from_numpy(
                    np.ascontiguousarray(X[start : start + self.batch_size])
                ).to(self.device)
                all_probs.append(torch.sigmoid(self._net(batch)).cpu().numpy())
        return np.concatenate(all_probs, axis=0)

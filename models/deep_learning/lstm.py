# Stacked bidirectional LSTM with attention pooling for flash crash prediction.
# Architecture: input -> BiLSTM (n_layers) -> attention pooling -> classification head.
# Attention pooling uses a learnable query vector to compute softmax weights over
# all LSTM hidden states and return a single context vector for classification.

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.base import BaseFlashCrashModel


class AttentionPooling(nn.Module):
    # A single learnable query vector computes compatibility with every hidden state,
    # applies softmax, and returns the weighted sum.

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.scale  = hidden_dim ** -0.5

    def forward(self, h):
        # h: (batch, seq_len, hidden_dim) -> context: (batch, hidden_dim)
        scores  = torch.einsum("bth,h->bt", h, self.query) * self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,bth->bh", weights, h)


class _LSTMNet(nn.Module):
    # Bidirectional LSTM + attention pooling used internally by LSTMFlashCrashModel.

    def __init__(self, n_features, hidden_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        lstm_out_dim   = hidden_dim * (2 if bidirectional else 1)
        self.attn_pool = AttentionPooling(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) -> logits: (batch,)
        h, _     = self.lstm(x)
        context  = self.attn_pool(h)
        return self.classifier(context).squeeze(-1)


class LSTMFlashCrashModel(BaseFlashCrashModel):

    name = "lstm"

    def __init__(self, seq_len, n_features, hidden_dim=128, n_layers=2, bidirectional=True,
                 dropout=0.2, lr=1e-3, epochs=50, batch_size=64, device="auto"):
        self.seq_len      = seq_len
        self.n_features   = n_features
        self.hidden_dim   = hidden_dim
        self.n_layers     = n_layers
        self.bidirectional = bidirectional
        self.dropout      = dropout
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self._net = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            checkpoint_dir=None, resume_from=None):
        print(f"[{self.name}] training on {self.device}  X={X_train.shape}  pos_frac={y_train.mean():.4f}")

        self._net = _LSTMNet(
            n_features=self.n_features, hidden_dim=self.hidden_dim,
            n_layers=self.n_layers, bidirectional=self.bidirectional, dropout=self.dropout,
        ).to(self.device)

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

        X_t    = torch.from_numpy(X_train.astype(np.float32))
        y_t    = torch.from_numpy(y_train.astype(np.float32))
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True, drop_last=False)

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

                if epoch % 10 == 0 or epoch == start_epoch:
                    msg = f"[{self.name}] epoch {epoch:>3}/{self.epochs}  loss={epoch_loss:.5f}"
                    if X_val is not None and y_val is not None:
                        val = self.evaluate(X_val, y_val)
                        msg += f"  val_auc={val['roc_auc']:.4f}  val_ap={val['avg_prec']:.4f}"
                    print(msg)

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
            raise RuntimeError("Initialise the network first by calling fit() once, "
                               "or use --resume in run.py which handles reconstruction automatically.")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._net.load_state_dict(ckpt["state_dict"])
        self._last_epoch = ckpt.get("epoch", 0)
        print(f"  [{self.name}] loaded from {path}  (epoch {self._last_epoch})")

    def predict_proba(self, X):
        if self._net is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        self._net.eval()
        X_t = torch.from_numpy(X.astype(np.float32))
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(X_t), self.batch_size):
                batch = X_t[start : start + self.batch_size].to(self.device)
                all_probs.append(torch.sigmoid(self._net(batch)).cpu().numpy())
        return np.concatenate(all_probs, axis=0)

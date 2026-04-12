# Transformer encoder with learnable positional encoding and CLS token pooling.
# Architecture: input_proj -> learnable positional encoding -> [CLS] token prepended
#               -> N TransformerEncoderLayers -> CLS output -> classification head.
# Using a CLS token avoids the information bottleneck of mean/max pooling and lets
# the model learn a global representation of the sequence.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.base import BaseFlashCrashModel


class LearnablePositionalEncoding(nn.Module):
    # Additive learnable positional encoding. +1 in max_len accounts for the CLS token.

    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class _TransformerNet(nn.Module):
    # Transformer encoder with CLS pooling used internally by TransformerFlashCrashModel.

    def __init__(self, seq_len, n_features, d_model, n_heads, n_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        # CLS token: one learnable vector broadcast over the batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = LearnablePositionalEncoding(seq_len, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,  # pre-LN for more stable training
            ),
            num_layers=n_layers,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: (B, seq_len, n_features) -> logits: (B,)
        B = x.shape[0]
        h   = self.input_proj(x)                         # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d_model)
        h   = self.pos_enc(torch.cat([cls, h], dim=1))   # (B, T+1, d_model)
        h   = self.encoder(h)
        return self.classifier(h[:, 0, :]).squeeze(-1)   # classify from CLS output at position 0


class TransformerFlashCrashModel(BaseFlashCrashModel):

    name = "transformer"

    def __init__(self, seq_len, n_features, d_model=128, n_heads=4, n_layers=3,
                 dim_feedforward=256, dropout=0.1, lr=1e-3, epochs=50, batch_size=64, device="auto"):
        self.seq_len         = seq_len
        self.n_features      = n_features
        self.d_model         = d_model
        self.n_heads         = n_heads
        self.n_layers        = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout         = dropout
        self.lr              = lr
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self._net = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"[{self.name}] training on {self.device}  X={X_train.shape}  pos_frac={y_train.mean():.4f}")

        self._net = _TransformerNet(
            seq_len=self.seq_len, n_features=self.n_features, d_model=self.d_model,
            n_heads=self.n_heads, n_layers=self.n_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        X_t    = torch.from_numpy(X_train.astype(np.float32))
        y_t    = torch.from_numpy(y_train.astype(np.float32))
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True, drop_last=False)

        self._net.train()
        for epoch in range(1, self.epochs + 1):
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

            if epoch % 10 == 0 or epoch == 1:
                msg = f"[{self.name}] epoch {epoch:>3}/{self.epochs}  loss={epoch_loss:.5f}"
                if X_val is not None and y_val is not None:
                    val = self.evaluate(X_val, y_val)
                    msg += f"  val_auc={val['roc_auc']:.4f}  val_ap={val['avg_prec']:.4f}"
                print(msg)

        return self

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

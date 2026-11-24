"""Quantile LSTM with additive temporal attention, trained from scratch on the pooled panel.

The architecture follows the author's Forecast_TimeSeries project (LSTM, softmax attention pooling over time, monotone
quantile head), reduced to a univariate input and the 9 quantile levels of the benchmark. Each window is standardised
with the mean and standard deviation of its own context, so that nothing outside the context is needed to forecast."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..config import LEVELS

CTX = 64


class QuantileLSTM(nn.Module):
    def __init__(self, hidden: int = 48, H: int = 5, K: int = len(LEVELS), drop: float = 0.2):
        super().__init__()
        self.H, self.K = H, K
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, H * K))

    def forward(self, x):  # x [B, CTX]
        out, _ = self.lstm(x.unsqueeze(-1))
        w = torch.softmax(torch.tanh(self.attn(out)).squeeze(-1), dim=1)
        raw = self.head((out * w.unsqueeze(-1)).sum(1)).view(-1, self.H, self.K)
        return torch.cat([raw[..., :1], raw[..., :1] + torch.cumsum(nn.functional.softplus(raw[..., 1:]), -1)], -1)


def _standardise(ctx: np.ndarray):
    c = ctx[:, -CTX:]
    m, s = c.mean(1, keepdims=True), c.std(1, keepdims=True) + 1e-8
    return ((c - m) / s).astype(np.float32), m.astype(np.float32), s.astype(np.float32)


class LSTMForecaster:
    """``fit`` on training windows (early stopping on validation windows), then ``quantiles`` like any other model."""

    def __init__(self, H: int, seed: int = 0):
        torch.manual_seed(seed)
        self.rng, self.H, self.net = np.random.default_rng(seed), H, QuantileLSTM(H=H)

    def _loss(self, x, y):
        e = y.unsqueeze(-1) - self.net(x)
        lv = torch.tensor(LEVELS, dtype=torch.float32)
        return torch.maximum(lv * e, (lv - 1) * e).mean()

    def fit(self, train, val, epochs: int = 40, batch: int = 256, lr: float = 1e-3, patience: int = 6):
        (cx, ty), (vx, vy) = train, val
        X, m, s = _standardise(cx)
        Y = (ty - m) / s
        VX, vm, vs = _standardise(vx)
        VX, VY = torch.from_numpy(VX), torch.from_numpy((vy - vm) / vs)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-3)
        best, best_state, bad, self.history = np.inf, None, 0, []
        for ep in range(epochs):
            self.net.train()
            perm = self.rng.permutation(len(X))
            for i in range(0, len(X), batch):
                b = perm[i : i + batch]
                opt.zero_grad()
                self._loss(torch.from_numpy(X[b]), torch.from_numpy(Y[b])).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(self._loss(VX, VY))
            self.history.append(v)
            if v < best - 1e-5:
                best, best_state, bad = v, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    @torch.no_grad()
    def quantiles(self, contexts: np.ndarray, H: int) -> np.ndarray:
        x, m, s = _standardise(contexts)
        q = self.net(torch.from_numpy(x)).numpy()[:, :H]
        return q * s[..., None] + m[..., None]

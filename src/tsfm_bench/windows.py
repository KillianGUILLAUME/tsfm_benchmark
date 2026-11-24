"""Training / validation windows shared by the models that are trained (LSTM, LoRA fine-tuning).

A window is (asset a, origin t): context = values[t-L+1 .. t], targets = values[t+1 .. t+H]. Windows are assigned to a
split by the dates of their *targets*: every target of a training window is < VAL_START, every target of a validation
window is in [VAL_START, TEST_START). No target of the test period is ever used, for training or for model selection."""
from __future__ import annotations

import numpy as np

from .config import CONTEXT, TEST_START, VAL_START
from .data import Panel

MIN_CONTEXT = 128  # shorter histories are left-padded with NaN up to the context length


def split_origins(dates, split: str, H: int, min_context: int = MIN_CONTEXT) -> np.ndarray:
    idx = np.arange(min_context - 1, len(dates) - H)
    first, last = dates[idx + 1], dates[idx + H]
    if split == "train":
        keep = last < VAL_START
    elif split == "val":
        keep = (first >= VAL_START) & (last < TEST_START)
    else:
        raise ValueError(split)
    return idx[np.asarray(keep)]


def make_windows(panel: Panel, task: str, split: str, H: int, length: int = CONTEXT):
    """Contexts [N, length] (NaN-padded) and targets [N, H] for every (origin, asset) of the split."""
    v = panel.values(task)
    origins = split_origins(panel.dates, split, H)
    ctx = np.full((len(origins) * v.shape[1], length), np.nan, dtype=np.float32)
    tgt = np.empty((len(origins) * v.shape[1], H), dtype=np.float32)
    n = 0
    for t in origins:
        lo = max(0, t - length + 1)
        for a in range(v.shape[1]):
            ctx[n, length - (t - lo + 1) :] = v[lo : t + 1, a]
            tgt[n] = v[t + 1 : t + H + 1, a]
            n += 1
    return ctx, tgt

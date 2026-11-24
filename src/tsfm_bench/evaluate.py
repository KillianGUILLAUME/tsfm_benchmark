"""Rolling-origin evaluation harness shared by every model.

Protocol: a forecast "at origin t" is made at the end of day t using data up to and including day t, for the value of
day t+h. Test origins are those whose target date is on/after ``TEST_START`` (and exists). Every model produces the
same 9 quantiles (levels 0.1 ... 0.9) so that all models are scored with the same rules on the same observations.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import CONTEXT, HORIZONS, LEVELS, PRED_DIR, QCOLS, TEST_START
from .data import Panel
from .models.classical import gaussian_quantiles


def test_origins(dates: pd.DatetimeIndex, h: int) -> np.ndarray:
    """Indices t such that day t+h exists and falls in the test period."""
    idx = np.arange(len(dates) - h)
    return idx[np.asarray(dates[idx + h] >= TEST_START)]


def build_contexts(values: np.ndarray, origins: np.ndarray, length: int = CONTEXT) -> np.ndarray:
    """History windows for foundation models: shape [len(origins) * A, length], origin-major.

    Row ``i * A + a`` is ``values[t-length+1 : t+1, a]`` for ``t = origins[i]`` (left-padded with NaN), i.e. it
    never contains anything after the origin."""
    T, A = values.shape
    padded = np.vstack([np.full((length - 1, A), np.nan), values]).astype(np.float32)
    out = np.empty((len(origins), A, length), dtype=np.float32)
    for a in range(A):
        out[:, a, :] = sliding_window_view(padded[:, a], length)[origins]
    return out.reshape(len(origins) * A, length)


def to_frame(panel: Panel, task: str, h: int, origins: np.ndarray, q: np.ndarray, model: str, seconds: float):
    """Long table of forecasts: one row per (asset, origin); q has shape [n_origins, A, K]."""
    n, A, _ = q.shape
    cols = {
        "model": model,
        "task": task,
        "h": h,
        "asset": np.tile(np.array(panel.assets), n),
        "origin": np.repeat(panel.dates[origins].values, A),
        "target": np.repeat(panel.dates[origins + h].values, A),
        "y": panel.values(task)[origins + h].reshape(-1),
        "seconds": seconds,
    }
    for k, name in enumerate(QCOLS):
        cols[name] = q[:, :, k].reshape(-1)
    return pd.DataFrame(cols)


def save_predictions(df: pd.DataFrame, task: str, model: str) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRED_DIR / f"{task}__{model}.parquet")


def run_vol_model(panel: Panel, model) -> pd.DataFrame:
    """Volatility task (log realized variance) for a classical point model with Gaussian predictive law."""
    t0 = time.time()
    pts = [model.point(panel.logrv[:, a], HORIZONS) for a in range(len(panel.assets))]
    seconds = time.time() - t0
    frames = []
    for h in HORIZONS:
        origins = test_origins(panel.dates, h)
        q = np.stack(
            [gaussian_quantiles(panel.logrv[:, a], pts[a][h], h)[origins] for a in range(len(panel.assets))], axis=1
        )
        frames.append(to_frame(panel, "logrv", h, origins, q, model.name, seconds))
    return pd.concat(frames, ignore_index=True)


def run_ret_model(panel: Panel, model) -> pd.DataFrame:
    """Returns task (h = 1) for a classical model producing quantiles directly."""
    t0 = time.time()
    qa = [model.quantiles(panel.ret[:, a]) for a in range(len(panel.assets))]
    seconds = time.time() - t0
    origins = test_origins(panel.dates, 1)
    q = np.stack([x[origins] for x in qa], axis=1)
    return to_frame(panel, "ret", 1, origins, q, model.name, seconds)


def run_foundation_model(panel: Panel, task: str, forecaster, name: str) -> pd.DataFrame:
    """Any model exposing ``quantiles(contexts, H) -> [N, H, K]`` (foundation models, LSTM...) on a task."""
    horizons = HORIZONS if task == "logrv" else (1,)
    hmax = max(horizons)
    first = test_origins(panel.dates, hmax)[0]
    union = np.arange(first, len(panel.dates) - 1)  # every origin needed by at least one horizon
    ctx = build_contexts(panel.values(task), union)
    t0 = time.time()
    qq = forecaster.quantiles(ctx, hmax)  # [n_union * A, hmax, K]
    seconds = time.time() - t0
    A = len(panel.assets)
    qq = qq.reshape(len(union), A, hmax, -1)
    frames = []
    for h in horizons:
        origins = test_origins(panel.dates, h)
        rows = np.searchsorted(union, origins)
        frames.append(to_frame(panel, task, h, origins, qq[rows, :, h - 1, :], name, seconds))
    return pd.concat(frames, ignore_index=True)

"""Data: public Binance hourly klines -> daily returns and daily log realized variance.

The two target series of the benchmark, for a day D (UTC):
  * ``ret``   : sum of the 24 hourly log-returns of D (daily log-return);
  * ``logrv`` : log of the sum of their squares (log realized variance), the standard proxy of daily volatility.
A day is only used if its 24 hourly bars are present and contiguous. Both values are known at the end of day D,
which is when a forecast "made at origin D" is produced.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, DATA_END, DATA_START, UNIVERSE

KLINES_URL = "https://data-api.binance.vision/api/v3/klines"  # public market-data endpoint, no API key
HOUR_MS = 3_600_000


def _get_klines(s: requests.Session, symbol: str, start_ms: int, retries: int = 6) -> list:
    """One page of klines, retried with exponential back-off on network errors / rate limits."""
    for k in range(retries):
        try:
            resp = s.get(
                KLINES_URL,
                params={"symbol": f"{symbol}USDT", "interval": "1h", "startTime": start_ms, "limit": 1000},
                timeout=30,
            )
            if resp.status_code in (418, 429) or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if k == retries - 1:
                raise
            time.sleep(2**k)
    return []


def fetch_hourly(symbol: str, start: str = DATA_START, session: requests.Session | None = None) -> pd.DataFrame:
    """Hourly closes of ``{symbol}USDT`` from ``start`` until now (the bar still forming is dropped)."""
    s = session or requests.Session()
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    rows, cur = [], start_ms
    while cur < now_ms:
        batch = _get_klines(s, symbol, cur)
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + HOUR_MS
        time.sleep(0.05)
    df = pd.DataFrame(rows).iloc[:, [0, 4, 6]]
    df.columns = ["open_time", "close", "close_time"]
    df = df.astype({"open_time": "int64", "close": "float64", "close_time": "int64"})
    df = df[df["close_time"] < now_ms]
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df.drop_duplicates("time").set_index("time")[["close"]]


def daily_from_hourly(h: pd.DataFrame) -> pd.DataFrame:
    """Daily ``ret`` and ``logrv`` from hourly closes.

    Short exchange outages (a few missing hourly bars) are bridged by carrying the last close forward (at most 6 hours,
    the cumulated move then shows up in the next bar); days still without 24 hourly returns are dropped."""
    h = h.sort_index()
    grid = pd.date_range(h.index[0], h.index[-1], freq="h")
    c = h["close"].reindex(grid).ffill(limit=6)
    r = np.log(c).diff()
    g = pd.DataFrame({"r": r.values, "r2": (r**2).values, "day": grid.floor("D")})
    agg = g.groupby("day").agg(ret=("r", "sum"), rv=("r2", "sum"), n=("r", "count"))
    agg = agg[(agg["n"] == 24) & (agg["rv"] > 0)].copy()
    agg["logrv"] = np.log(agg["rv"])
    return agg[["ret", "logrv"]]


def build_daily_file(path: Path | None = None, assets: list[str] | None = None) -> pd.DataFrame:
    """Download every asset, build the daily table (long format) and save it as parquet."""
    path = path or DATA_DIR / "daily.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with requests.Session() as s:
        for a in assets or UNIVERSE:
            cache = DATA_DIR / "hourly" / f"{a}.parquet"
            if cache.exists():
                hourly = pd.read_parquet(cache)
            else:
                hourly = fetch_hourly(a, session=s)
                cache.parent.mkdir(parents=True, exist_ok=True)
                hourly.to_parquet(cache)
            d = daily_from_hourly(hourly)
            d["asset"] = a
            frames.append(d.reset_index())
            print(f"{a}: {len(d)} days, {d.index.min().date()} -> {d.index.max().date()}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(path)
    return out


@dataclass
class Panel:
    """Aligned daily series. Row t of every array is day ``dates[t]``; assets are the columns."""

    dates: pd.DatetimeIndex
    assets: list[str]
    ret: np.ndarray
    logrv: np.ndarray

    def values(self, task: str) -> np.ndarray:
        return {"ret": self.ret, "logrv": self.logrv}[task]


def load_panel(path: Path | None = None) -> Panel:
    """Load the daily file and keep the longest contiguous calendar on which every asset is present."""
    path = path or DATA_DIR / "daily.parquet"
    long = pd.read_parquet(path)
    long = long[long["day"] <= DATA_END]
    ret = long.pivot(index="day", columns="asset", values="ret")
    lrv = long.pivot(index="day", columns="asset", values="logrv")
    ok = ret.notna().all(axis=1) & lrv.notna().all(axis=1)
    ret, lrv = ret[ok], lrv[ok]
    # keep the longest run of consecutive days (a hole would make neighbouring rows non-adjacent in time)
    d = ret.index.to_series().diff() != pd.Timedelta(days=1)
    run_id = d.cumsum()
    best = run_id.value_counts().idxmax()
    keep = (run_id == best).values
    ret, lrv = ret[keep], lrv[keep]
    assets = [a for a in UNIVERSE if a in ret.columns]
    return Panel(pd.DatetimeIndex(ret.index), assets, ret[assets].to_numpy(), lrv[assets].to_numpy())

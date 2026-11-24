"""Look-ahead tests: replacing everything after a date by garbage must not change a forecast made at or before it."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_bench.config import LEVELS, TEST_START, VAL_START  # noqa: E402
from tsfm_bench.data import Panel, daily_from_hourly  # noqa: E402
from tsfm_bench.evaluate import build_contexts, test_origins as origins_of_test  # noqa: E402
from tsfm_bench.metrics import calibration, central_coverage, dm_test, pinball  # noqa: E402
from tsfm_bench.models.classical import RET_MODELS, VOL_MODELS, gaussian_quantiles  # noqa: E402
from tsfm_bench.windows import make_windows, split_origins  # noqa: E402

T, CUT, HZ = 480, 400, (1, 5)


def logrv_like(seed=0):
    rng = np.random.default_rng(seed)
    y = np.empty(T)
    y[0] = -7.0
    for t in range(1, T):
        y[t] = -7.0 + 0.8 * (y[t - 1] + 7.0) + 0.6 * rng.standard_normal()
    return y


def garbage_after(y, scale, seed=99):
    z = y.copy()
    z[CUT + 1 :] = scale * np.random.default_rng(seed).standard_normal(T - CUT - 1)
    return z


def check_point(model):
    y = logrv_like()
    a, b = model.point(y, HZ), model.point(garbage_after(y, 20.0), HZ)
    for h in HZ:
        np.testing.assert_allclose(a[h][: CUT + 1], b[h][: CUT + 1], rtol=1e-9, atol=1e-9, equal_nan=True)
        assert np.isfinite(a[h][300 : CUT + 1]).all()


@pytest.mark.parametrize("model", VOL_MODELS, ids=lambda m: m.name)
def test_vol_models_ignore_the_future(model):
    check_point(model)


@pytest.mark.parametrize("model", RET_MODELS, ids=lambda m: m.name)
def test_return_models_ignore_the_future(model):
    r = 0.03 * np.random.default_rng(1).standard_t(5, T)
    a, b = model.quantiles(r), model.quantiles(garbage_after(r, 0.5))
    np.testing.assert_allclose(a[: CUT + 1], b[: CUT + 1], rtol=1e-9, atol=1e-12, equal_nan=True)


def test_negative_control_a_leaky_model_is_caught():
    class Leaky:
        def point(self, y, horizons):
            return {h: (y - y.mean()) / y.std() for h in horizons}  # full-sample standardisation: classic leak

    with pytest.raises(AssertionError):
        check_point(Leaky())


@pytest.mark.parametrize("h", HZ)
def test_predictive_scale_only_uses_realised_errors(h):
    y = logrv_like()
    f = np.r_[y[0], y[:-1]]
    a, b = gaussian_quantiles(y, f, h), gaussian_quantiles(garbage_after(y, 20.0), f, h)
    np.testing.assert_allclose(a[: CUT + 1], b[: CUT + 1], equal_nan=True)


def test_contexts_stop_at_the_origin():
    v = np.arange(2 * T, dtype=float).reshape(T, 2)
    c = build_contexts(v, np.array([CUT]), length=32)
    np.testing.assert_array_equal(c, v[CUT - 31 : CUT + 1].T)
    w = v.copy()
    w[CUT + 1 :] = -1e9
    np.testing.assert_array_equal(c, build_contexts(w, np.array([CUT]), length=32))


def make_panel():
    dates = pd.date_range("2021-01-02", "2026-11-23", freq="D")
    x = np.random.default_rng(0).standard_normal((len(dates), 3))
    return Panel(dates, ["A", "B", "C"], x, x)


@pytest.mark.parametrize("H", HZ)
def test_splits_are_disjoint_and_never_touch_the_test_period(H):
    p = make_panel()
    tr, va, te = split_origins(p.dates, "train", H), split_origins(p.dates, "val", H), origins_of_test(p.dates, H)
    assert p.dates[tr.max() + H] < VAL_START <= p.dates[va.min() + 1]
    assert p.dates[va.max() + H] < TEST_START <= p.dates[te.min() + H]


def test_windows_content():
    p = make_panel()
    ctx, tgt = make_windows(p, "ret", "val", 5, length=64)
    t = split_origins(p.dates, "val", 5)[0]
    np.testing.assert_allclose(ctx[1], p.ret[t - 63 : t + 1, 1], rtol=1e-6)
    np.testing.assert_allclose(tgt[1], p.ret[t + 1 : t + 6, 1], rtol=1e-6)


def test_daily_aggregation_drops_incomplete_days_and_uses_one_day_only():
    idx = pd.date_range("2024-01-01", periods=24 * 4 - 3, freq="h")
    c = pd.DataFrame({"close": np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(idx))))}, index=idx)
    d = daily_from_hourly(c)
    assert list(d.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))  # first and last days are incomplete
    r = np.log(c["close"]).diff()["2024-01-02"]
    assert d["ret"].iloc[0] == pytest.approx(r.sum())
    assert d["logrv"].iloc[0] == pytest.approx(np.log((r**2).sum()))


def test_metrics_on_known_values():
    y = np.random.default_rng(0).standard_normal(200_000)
    from scipy.stats import norm

    good = np.tile(norm.ppf(LEVELS), (len(y), 1))
    assert central_coverage(y, good) == pytest.approx(0.8, abs=0.005)
    assert np.abs(calibration(y, good) - LEVELS).max() < 0.005
    assert pinball(y, good).mean() < pinball(y, 1.5 * good).mean()
    assert pinball(np.array([1.0]), np.zeros((1, 9)))[0] == pytest.approx(LEVELS.mean())
    d = np.random.default_rng(1).standard_normal(5000)
    assert dm_test(d, np.zeros(5000))[1] > 0.01 and dm_test(d + 0.2, np.zeros(5000))[1] < 1e-6

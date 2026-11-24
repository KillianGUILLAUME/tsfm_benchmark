"""Classical baselines. Every model is *causal* by construction: the forecast made at origin t only uses y[:t+1].

* Volatility task (target: log realized variance) -- point forecasts f[h][t] of y[t+h]:
  Naive, EWMA, HAR-RV, ARIMA(1,0,1), local-level state-space model (Kalman filter).
  Their predictive distribution is Gaussian, centred on the point forecast, with a scale given by the RMS of the
  errors *already realised* at the origin (see ``gaussian_quantiles``).
* Returns task (target: daily log-return, h = 1) -- direct quantile forecasts:
  GARCH(1,1) with Student-t innovations, EWMA volatility, historical simulation.

Parameters are re-estimated every ``REFIT_EVERY`` origins on data available at that date (expanding window).
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm, t as student_t

from ..config import ERR_WINDOW, FIRST_FIT, LEVELS, MIN_ERR_OBS, REFIT_EVERY

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------
# Gaussian predictive law from point forecasts
# ------------------------------------------------------------------------------------------------
def gaussian_quantiles(y: np.ndarray, f: np.ndarray, h: int, window: int = ERR_WINDOW, min_obs: int = MIN_ERR_OBS):
    """Quantiles ``f[t] + sigma_t * z`` of the h-step forecast made at t.

    ``sigma_t`` is the RMS of the errors ``y[s+h] - f[s]`` over the last ``window`` origins ``s <= t-h``: these are
    exactly the errors that are already observable at time t (the error of origin s is realised at s+h).
    Returns an array [T, K]; rows without enough history are NaN."""
    T = len(y)
    e = np.full(T, np.nan)
    e[: T - h] = y[h:] - f[: T - h]
    e2 = np.nan_to_num(e**2)
    valid = (~np.isnan(e)).astype(float)
    cs, cn = np.concatenate([[0.0], np.cumsum(e2)]), np.concatenate([[0.0], np.cumsum(valid)])
    sigma = np.full(T, np.nan)
    for t in range(h, T):
        hi = t - h + 1  # exclusive upper bound -> origins s <= t-h
        lo = max(0, hi - window)
        n = cn[hi] - cn[lo]
        if n >= min_obs:
            sigma[t] = np.sqrt((cs[hi] - cs[lo]) / n)
    return f[:, None] + sigma[:, None] * norm.ppf(LEVELS)[None, :]


# ------------------------------------------------------------------------------------------------
# Volatility task: point models  (point(y, horizons) -> {h: f}, f[t] forecasts y[t+h] from y[:t+1])
# ------------------------------------------------------------------------------------------------
class Naive:
    name = "naive"

    def point(self, y, horizons):
        return {h: y.copy() for h in horizons}


class EWMA:
    """Exponentially weighted moving average of the series (RiskMetrics decay)."""

    def __init__(self, lam: float = 0.94):
        self.lam, self.name = lam, "ewma"

    def point(self, y, horizons):
        s = np.empty_like(y)
        s[0] = y[0]
        for t in range(1, len(y)):
            s[t] = self.lam * s[t - 1] + (1 - self.lam) * y[t]
        return {h: s.copy() for h in horizons}


class HAR:
    """HAR-RV (Corsi, 2009) on log realized variance: regress y[t+h] on y[t], mean(y[t-4..t]), mean(y[t-21..t]).

    OLS on an expanding window; the pairs used at a refit date t0 are those whose target is already realised
    (targets y[s+h] with s + h <= t0)."""

    def __init__(self, weekday: bool = False):
        # weekday=True adds day-of-week dummies of the *target* day (known in advance): crypto volatility drops on
        # week-ends. The calendar is contiguous (see load_panel), so the weekday is the row index modulo 7.
        self.weekday, self.name = weekday, "har_dow" if weekday else "har"

    def point(self, y, horizons):
        T = len(y)
        c = np.concatenate([[0.0], np.cumsum(y)])
        m5, m22 = np.full(T, np.nan), np.full(T, np.nan)
        m5[4:] = (c[5:] - c[:-5]) / 5
        m22[21:] = (c[22:] - c[:-22]) / 22
        X = np.column_stack([np.ones(T), y, m5, m22])
        out = {}
        for h in horizons:
            f = np.full(T, np.nan)
            Z = X
            if self.weekday:
                Z = np.column_stack([X, np.eye(7)[(np.arange(T) + h) % 7][:, 1:]])
            for t0 in range(FIRST_FIT, T, REFIT_EVERY):
                idx = np.arange(21, t0 - h + 1)
                beta, *_ = np.linalg.lstsq(Z[idx], y[idx + h], rcond=None)
                end = min(t0 + REFIT_EVERY, T)
                f[t0:end] = Z[t0:end] @ beta
            out[h] = f
        return out


class ARIMA101:
    """ARIMA(1,0,1) with constant, re-estimated monthly; the filter is extended one observation at a time."""

    name = "arima"

    def point(self, y, horizons):
        from statsmodels.tsa.arima.model import ARIMA

        T, H = len(y), max(horizons)
        out = {h: np.full(T, np.nan) for h in horizons}
        for t0 in range(FIRST_FIT, T, REFIT_EVERY):
            res = ARIMA(y[: t0 + 1], order=(1, 0, 1), trend="c").fit()
            for t in range(t0, min(t0 + REFIT_EVERY, T)):
                if t > t0:
                    res = res.extend(y[t : t + 1])
                fc = np.asarray(res.forecast(H))
                for h in horizons:
                    out[h][t] = fc[h - 1]
        return out


def kalman_local_level(y: np.ndarray, s2_eps: float, s2_eta: float) -> np.ndarray:
    """Filtered level a[t | t] of y[t] = mu[t] + eps, mu[t+1] = mu[t] + eta (causal forward recursion)."""
    a, P = y[0], 1e6  # diffuse start
    out = np.empty_like(y)
    for t in range(len(y)):
        P_pred = P + s2_eta if t > 0 else P
        F = P_pred + s2_eps
        K = P_pred / F
        a = a + K * (y[t] - a)
        P = P_pred * (1 - K)
        out[t] = a
    return out


class LocalLevel:
    """Local-level state-space model: variances estimated by maximum likelihood at each refit, Kalman filtering."""

    name = "local_level"

    def point(self, y, horizons):
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        T = len(y)
        f = np.full(T, np.nan)
        for t0 in range(FIRST_FIT, T, REFIT_EVERY):
            res = UnobservedComponents(y[: t0 + 1], level="llevel").fit(disp=False, maxiter=200)
            s2_eps, s2_eta = np.asarray(res.params)[:2]
            end = min(t0 + REFIT_EVERY, T)
            f[t0:end] = kalman_local_level(y[:end], s2_eps, s2_eta)[t0:end]  # filtered level only uses y[:t+1]
        return {h: f.copy() for h in horizons}


VOL_MODELS = [Naive(), EWMA(), HAR(), HAR(weekday=True), ARIMA101(), LocalLevel()]


# ------------------------------------------------------------------------------------------------
# Returns task: direct quantile models  (quantiles(r) -> [T, K], row t forecasts r[t+1] from r[:t+1])
# ------------------------------------------------------------------------------------------------
class GarchT:
    """Constant-mean GARCH(1,1) with Student-t innovations (arch), parameters re-estimated monthly."""

    name = "garch_t"

    def quantiles(self, r):
        from arch import arch_model

        T = len(r)
        x = 100.0 * r
        q = np.full((T, len(LEVELS)), np.nan)
        spec = dict(mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        for t0 in range(FIRST_FIT, T, REFIT_EVERY):
            params = arch_model(x[: t0 + 1], **spec).fit(disp="off", show_warning=False).params
            fixed = arch_model(x, **spec).fix(params)
            fc = fixed.forecast(horizon=1, start=t0, reindex=False)  # origin s uses x[:s+1] only
            n = min(REFIT_EVERY, T - t0)
            mu = fc.mean.values[:n, 0] / 100.0
            sigma = np.sqrt(fc.variance.values[:n, 0]) / 100.0
            nu = float(params["nu"])
            z = student_t.ppf(LEVELS, nu) * np.sqrt((nu - 2.0) / nu)  # unit-variance Student-t quantiles
            q[t0 : t0 + n] = mu[:, None] + sigma[:, None] * z[None, :]
        return q


class EwmaVol:
    """Zero-mean Gaussian with RiskMetrics EWMA variance."""

    def __init__(self, lam: float = 0.94):
        self.lam, self.name = lam, "ewma_vol"

    def quantiles(self, r):
        T = len(r)
        var = np.full(T, np.nan)
        v = np.var(r[:30])
        for t in range(T):
            v = self.lam * v + (1 - self.lam) * r[t] ** 2
            var[t] = v
        return np.sqrt(var)[:, None] * norm.ppf(LEVELS)[None, :]


class HistSim:
    """Historical simulation: empirical quantiles of the last ``window`` returns."""

    def __init__(self, window: int = 250):
        self.window, self.name = window, "hist_sim"

    def quantiles(self, r):
        T = len(r)
        q = np.full((T, len(LEVELS)), np.nan)
        w = sliding_window_view(r, self.window)  # w[j] = r[j : j+window] -> available at t = j+window-1
        q[self.window - 1 :] = np.quantile(w, LEVELS, axis=1).T
        return q


RET_MODELS = [GarchT(), EwmaVol(), HistSim()]

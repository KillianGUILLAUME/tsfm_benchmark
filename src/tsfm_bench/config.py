"""Experiment configuration: everything that defines the evaluation protocol lives here."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"

# 12 liquid assets taken from the universe of the pairs-trading project (all listed before 2021)
UNIVERSE = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "LINK", "LTC", "TRX", "AVAX", "UNI"]
DATA_START = "2021-01-01"
DATA_END = pd.Timestamp("2026-11-23")   # last day used: keeps the results reproducible when data is re-downloaded

# ---- time split ------------------------------------------------------------------------------
# Targets dated on/after TEST_START form the test set. Chronos-Bolt was released in late 2024, so a test
# period starting in 2025 limits the risk that its pre-training corpus contains the evaluated prices.
TEST_START = pd.Timestamp("2025-01-01")
# LoRA fine-tuning: training targets < VAL_START, early-stopping targets in [VAL_START, TEST_START).
VAL_START = pd.Timestamp("2024-01-01")

# ---- forecasting task ------------------------------------------------------------------------
HORIZONS = (1, 5)                       # days ahead (volatility task); the returns task uses h = 1
LEVELS = np.round(np.arange(0.1, 1.0, 0.1), 1)   # the 9 quantile levels natively produced by Chronos-Bolt
QCOLS = [f"q{int(round(l * 100))}" for l in LEVELS]
CONTEXT = 512                           # days of history given to the foundation models

# ---- classical models ------------------------------------------------------------------------
FIRST_FIT = 250                         # first origin at which classical models are estimated
REFIT_EVERY = 21                        # re-estimation period, in origins (~ monthly)
ERR_WINDOW = 250                        # past realised errors used to scale Gaussian predictive laws
MIN_ERR_OBS = 60

# reference model for the skill scores of each task
REFERENCE = {"logrv": "har_dow", "ret": "garch_t"}

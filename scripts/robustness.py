"""Robustness: DM tests between chosen pairs of models, per-asset wins and per-year losses."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_bench.config import QCOLS  # noqa: E402
from tsfm_bench.metrics import dm_test, pinball  # noqa: E402
from tsfm_bench.report import align, load_all  # noqa: E402

d = align(load_all())
d["loss"] = pinball(d["y"].to_numpy(), d[QCOLS].to_numpy())
PAIRS = {"logrv": [("chronos_bolt_small", "har"), ("chronos_bolt_small", "har_dow"), ("chronos_bolt_small_lora", "chronos_bolt_small"),
                   ("chronos_bolt_small_lora", "har_dow"), ("chronos_bolt_small", "lstm_attn")],
         "ret": [("chronos_bolt_small", "garch_t"), ("chronos_bolt_small_lora", "chronos_bolt_small"), ("chronos_bolt_small_lora", "garch_t")]}
for (task, h), g in d.groupby(["task", "h"]):
    by_date = g.pivot_table(index="origin", columns="model", values="loss")
    by_asset = g.pivot_table(index="asset", columns="model", values="loss")
    print(f"== {task} h={h}  origins={len(by_date)} assets={len(by_asset)}")
    for a, b in PAIRS[task]:
        if a in by_date and b in by_date:
            _, p = dm_test(by_date[a].to_numpy(), by_date[b].to_numpy())
            print(f"  {a} vs {b}: loss {100 * (by_date[a].mean() / by_date[b].mean() - 1):+.2f}%  DM p={p:.4f}  "
                  f"better on {int((by_asset[a] < by_asset[b]).sum())}/{len(by_asset)} assets")
    y = g.assign(year=pd.to_datetime(g["target"]).dt.year).pivot_table(index="year", columns="model", values="loss")
    print(y.round(5).T.to_string())

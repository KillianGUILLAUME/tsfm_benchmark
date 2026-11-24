"""Run the classical baselines on both tasks and save their test-period quantile forecasts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_bench.data import load_panel  # noqa: E402
from tsfm_bench.evaluate import run_ret_model, run_vol_model, save_predictions  # noqa: E402
from tsfm_bench.models.classical import RET_MODELS, VOL_MODELS  # noqa: E402

if __name__ == "__main__":
    panel = load_panel()
    only = set(sys.argv[1:])
    for m in VOL_MODELS:
        if only and m.name not in only:
            continue
        df = run_vol_model(panel, m)
        save_predictions(df, "logrv", m.name)
        print(f"logrv  {m.name:<12} rows={len(df):>6}  {df['seconds'].iloc[0]:6.1f}s", flush=True)
    for m in RET_MODELS:
        if only and m.name not in only:
            continue
        df = run_ret_model(panel, m)
        save_predictions(df, "ret", m.name)
        print(f"ret    {m.name:<12} rows={len(df):>6}  {df['seconds'].iloc[0]:6.1f}s", flush=True)

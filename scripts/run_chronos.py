"""Forecast the test period with a Chronos-Bolt checkpoint: run_chronos.py SIZE TASK [ADAPTER_DIR NAME]."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HOME=str(ROOT / ".hf_cache"), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
sys.path.insert(0, str(ROOT / "src"))

from tsfm_bench.config import QCOLS  # noqa: E402
from tsfm_bench.data import load_panel  # noqa: E402
from tsfm_bench.evaluate import run_foundation_model, save_predictions  # noqa: E402
from tsfm_bench.metrics import pinball  # noqa: E402
from tsfm_bench.models.chronos_models import ChronosForecaster  # noqa: E402

if __name__ == "__main__":
    size, task = sys.argv[1:3]
    adapter, name = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else (None, f"chronos_bolt_{size}")
    f = ChronosForecaster(str(ROOT / ".hf_cache" / f"chronos-bolt-{size}"), adapter)
    df = run_foundation_model(load_panel(), task, f, name)
    save_predictions(df, task, name)
    print(task, name, {h: round(pinball(g["y"].to_numpy(), g[QCOLS].to_numpy()).mean(), 5) for h, g in df.groupby("h")},
          f"{df['seconds'].iloc[0]:.0f}s", flush=True)

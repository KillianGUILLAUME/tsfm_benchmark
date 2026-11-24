"""Train the attention LSTM on targets < VAL_START (early stopping on 2024) and forecast the test period."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_bench.config import QCOLS  # noqa: E402
from tsfm_bench.data import load_panel  # noqa: E402
from tsfm_bench.evaluate import run_foundation_model, save_predictions  # noqa: E402
from tsfm_bench.metrics import pinball  # noqa: E402
from tsfm_bench.models.lstm import LSTMForecaster  # noqa: E402
from tsfm_bench.windows import make_windows  # noqa: E402

if __name__ == "__main__":
    panel = load_panel()
    for task, H in (("logrv", 5), ("ret", 1)):
        train, val = make_windows(panel, task, "train", H), make_windows(panel, task, "val", H)
        for seed in (0, 1, 2):
            m = LSTMForecaster(H, seed).fit(train, val)
            df = run_foundation_model(panel, task, m, "lstm_attn")
            s = {h: pinball(g["y"].to_numpy(), g[QCOLS].to_numpy()).mean() for h, g in df.groupby("h")}
            print(task, "seed", seed, "windows", len(train[0]), len(val[0]), "epochs", len(m.history),
                  "best val", round(min(m.history), 4), "test pinball", {h: round(x, 5) for h, x in s.items()}, flush=True)
            if seed == 0:  # fixed in advance: seeds 1-2 only show the dispersion, no selection on the test set
                save_predictions(df, task, "lstm_attn")

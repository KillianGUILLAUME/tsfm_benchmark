"""Download hourly Binance klines and build data/daily.parquet (daily returns and log realized variance)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_bench.data import build_daily_file  # noqa: E402

if __name__ == "__main__":
    out = build_daily_file()
    print(out.shape)

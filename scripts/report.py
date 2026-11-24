"""Score all saved predictions and write results/summary.md, tables and figures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_bench.report import main  # noqa: E402

if __name__ == "__main__":
    main()

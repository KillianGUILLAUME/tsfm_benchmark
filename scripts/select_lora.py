"""Pick, for each task, the learning rate with the lowest 2024 validation loss (the test period is not read)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
task = sys.argv[1]
runs = {p.parent: json.loads(p.read_text()) for p in sorted((ROOT / "results/adapters").glob(f"small_{task}_lr*/log.json"))}
best = min(runs, key=lambda d: min(v for _, v in runs[d]["val"]))
for d, r in runs.items():
    print(d.name, "zero-shot", round(r["val"][0][1], 5), "best", round(min(v for _, v in r["val"]), 5), "@", r["best_step"], file=sys.stderr)
print(best)

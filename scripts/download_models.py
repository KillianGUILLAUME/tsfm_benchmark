"""Download the two Chronos-Bolt checkpoints used here (weights + config only, anonymous, no telemetry)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HOME=str(ROOT / ".hf_cache"), HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")

from huggingface_hub import snapshot_download  # noqa: E402

REVISIONS = {"tiny": "a0e552de83495b5c28c14c71c374f3e33280b340", "small": "772f3d25d38aec6d914c8949dab4462e2d46f5d8"}

for size, revision in REVISIONS.items():
    path = snapshot_download(f"amazon/chronos-bolt-{size}", allow_patterns=["model.safetensors", "config.json"],
                             token=False, revision=revision, local_dir=ROOT / ".hf_cache" / f"chronos-bolt-{size}")
    print(size, path)

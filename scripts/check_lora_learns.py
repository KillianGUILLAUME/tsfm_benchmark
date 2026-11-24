"""Sanity check: the LoRA loop must be able to overfit one fixed training batch (gradients flow, loss is the right one)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HOME=str(ROOT / ".hf_cache"), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from finetune_lora import norm_pinball  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from tsfm_bench.data import load_panel  # noqa: E402
from tsfm_bench.models.chronos_models import ChronosForecaster  # noqa: E402
from tsfm_bench.windows import make_windows  # noqa: E402

torch.manual_seed(0)
cx, ty = make_windows(load_panel(), "logrv", "train", 5)
b = np.random.default_rng(0).integers(0, len(cx), 32)
net = ChronosForecaster(str(ROOT / ".hf_cache" / "chronos-bolt-small")).pipe.model
net = get_peft_model(net, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q", "v"]))
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
net.eval()  # no dropout: same function at every step
for k in range(41):
    loss = norm_pinball(net, cx[b], ty[b], 5)
    if k % 10 == 0:
        print("step", k, "loss on the fixed batch", round(loss.item(), 4), flush=True)
    opt.zero_grad()
    loss.backward()
    opt.step()

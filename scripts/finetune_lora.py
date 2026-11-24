"""LoRA fine-tuning of Chronos-Bolt: finetune_lora.py SIZE TASK STEPS LR.

Targets of the training windows are < VAL_START; the checkpoint is selected on 2024 targets only (step 0 = zero-shot is a
candidate, so fine-tuning can be rejected). The test period (>= TEST_START) is never read here."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HOME=str(ROOT / ".hf_cache"), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from tsfm_bench.config import RESULTS_DIR  # noqa: E402
from tsfm_bench.data import load_panel  # noqa: E402
from tsfm_bench.models.chronos_models import ChronosForecaster  # noqa: E402
from tsfm_bench.windows import make_windows  # noqa: E402


def norm_pinball(model, ctx, tgt, H):
    """Quantile loss on the H target steps, in units of each context's standard deviation (as in training)."""
    x, y = torch.from_numpy(ctx), torch.from_numpy(tgt)
    q = model(context=x).quantile_preds[:, :, :H]  # [B, K, H]
    s = torch.from_numpy(np.nanstd(ctx, axis=1, keepdims=True).astype(np.float32)) + 1e-8
    e = (y.unsqueeze(1) - q) / s.unsqueeze(1)
    lv = torch.arange(1, 10).float().view(1, 9, 1) / 10
    return torch.maximum(lv * e, (lv - 1) * e).mean()


if __name__ == "__main__":
    size, task, steps, lr = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    torch.manual_seed(0)
    rng, H, B = np.random.default_rng(0), 5 if task == "logrv" else 1, 32
    panel = load_panel()
    (cx, ty), (vx, vy) = make_windows(panel, task, "train", H), make_windows(panel, task, "val", H)
    pick = np.sort(rng.choice(len(vx), 1024, replace=False))
    vx, vy = vx[pick], vy[pick]
    net = ChronosForecaster(str(ROOT / ".hf_cache" / f"chronos-bolt-{size}")).pipe.model
    net = get_peft_model(net, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q", "v"]))
    net.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    out = RESULTS_DIR / "adapters" / f"{size}_{task}_lr{lr:g}"

    def validate():
        net.eval()
        with torch.no_grad():
            return float(np.mean([norm_pinball(net, vx[i : i + 128], vy[i : i + 128], H).item() for i in range(0, len(vx), 128)]))

    best = validate()
    log, best_step = [(0, best)], 0
    net.save_pretrained(out)
    print("step 0 (zero-shot) val", round(best, 5), flush=True)
    t0 = time.time()
    for step in range(1, steps + 1):
        net.train()
        b = rng.integers(0, len(cx), B)
        loss = norm_pinball(net, cx[b], ty[b], H)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            v = validate()
            log.append((step, v))
            if v < best:
                best, best_step = v, step
                net.save_pretrained(out)
            print(f"step {step} val {v:.5f} best {best:.5f}@{best_step} {time.time() - t0:.0f}s", flush=True)
    (out / "log.json").write_text(json.dumps({"lr": lr, "val": log, "best_step": best_step, "seconds": time.time() - t0}))

"""Chronos-Bolt wrapper (zero-shot or with a LoRA adapter), producing the 9 native quantiles.

Chronos-Bolt (Ansari et al., 2024) is a T5-based patched encoder-decoder pre-trained on a large corpus of time
series; it directly outputs quantiles for up to 64 future steps. It is used here as a pure function of the context
window: no sampling, no state, hence no way to look into the future if the context is built correctly."""
from __future__ import annotations

import numpy as np
import torch

from ..config import LEVELS


class ChronosForecaster:
    def __init__(self, model_id: str = "amazon/chronos-bolt-small", adapter: str | None = None, batch_size: int = 64):
        from chronos import BaseChronosPipeline

        self.pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
        if adapter is not None:
            from peft import PeftModel

            self.pipe.model = PeftModel.from_pretrained(self.pipe.model, adapter).merge_and_unload()
        self.pipe.model.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def quantiles(self, contexts: np.ndarray, H: int) -> np.ndarray:
        """contexts: [N, L] float32 (NaN = missing) -> quantile forecasts [N, H, K]."""
        out = []
        for i in range(0, len(contexts), self.batch_size):
            x = torch.from_numpy(np.ascontiguousarray(contexts[i : i + self.batch_size]))
            q, _ = self.pipe.predict_quantiles(x, prediction_length=H, quantile_levels=[float(l) for l in LEVELS])
            out.append(q.numpy())
        return np.concatenate(out, axis=0)

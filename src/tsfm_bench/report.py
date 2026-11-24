"""Score every saved prediction file; write tables (CSV + Markdown) and figures.

All models of a (task, horizon) pair are compared on exactly the same (asset, origin) observations. Scores:
  * pinball  : pinball loss averaged over the 9 quantile levels, the observations and the assets (lower is better);
  * skill    : 1 - pinball / pinball(reference model), so +10% means a 10% lower loss than the reference;
  * DM p     : Diebold-Mariano p-value against the reference (losses averaged over assets for each origin date,
               Newey-West variance), a small p-value means the difference is unlikely to be noise;
  * cov80    : share of observations inside the central 80% interval (nominal 0.80);
  * calib    : mean absolute gap between nominal and empirical frequency of the 9 quantiles;
  * point    : MAE of the median forecast (volatility task) / out-of-sample R^2 of the median vs a zero forecast (returns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LEVELS, PRED_DIR, QCOLS, REFERENCE, RESULTS_DIR
from .metrics import calibration, central_coverage, dm_test, pinball

TASK_LABEL = {"logrv": "Daily log realized variance", "ret": "Daily log-return"}


def family(model: str) -> str:
    if "lora" in model:
        return "foundation + LoRA"
    if model.startswith("chronos"):
        return "foundation zero-shot"
    if model.startswith("lstm"):
        return "trained from scratch"
    return "classical"


def load_all() -> pd.DataFrame:
    files = sorted(PRED_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no predictions in {PRED_DIR}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def align(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the (task, h, asset, origin) keys on which every model of the (task, h) group has a forecast."""
    out = []
    for (task, h), g in df.groupby(["task", "h"]):
        g = g.dropna(subset=QCOLS + ["y"])
        n_models = g["model"].nunique()
        counts = g.groupby(["asset", "origin"])["model"].transform("nunique")
        kept = g[counts == n_models].sort_values(["model", "asset", "origin"])
        out.append(kept)
    return pd.concat(out, ignore_index=True)


def score(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, h), g in df.groupby(["task", "h"]):
        ref_name = REFERENCE[task]
        per_model = {}
        for model, gm in g.groupby("model"):
            gm = gm.sort_values(["asset", "origin"])
            y, Q = gm["y"].to_numpy(), gm[QCOLS].to_numpy()
            loss = pinball(y, Q)
            per_model[model] = (gm, y, Q, loss)
        ref_loss = per_model[ref_name][3] if ref_name in per_model else None
        for model, (gm, y, Q, loss) in per_model.items():
            by_asset = pd.Series(loss, index=gm["asset"].to_numpy()).groupby(level=0).mean()
            row = {
                "task": task,
                "h": h,
                "model": model,
                "family": family(model),
                "n": len(y),
                "pinball": by_asset.mean(),
                "cov80": central_coverage(y, Q),
                "calib": float(np.abs(calibration(y, Q) - LEVELS).mean()),
                "seconds": float(gm["seconds"].iloc[0]),
            }
            med = Q[:, len(LEVELS) // 2]
            row["point"] = (
                float(np.mean(np.abs(y - med)))
                if task == "logrv"
                else float(1.0 - np.sum((y - med) ** 2) / np.sum(y**2))
            )
            if ref_loss is not None:
                row["skill_vs_ref"] = 1.0 - row["pinball"] / pd.Series(ref_loss, index=gm["asset"].to_numpy()).groupby(level=0).mean().mean()
                if model != ref_name:
                    d_model = pd.Series(loss, index=pd.MultiIndex.from_arrays([gm["origin"], gm["asset"]])).groupby(level=0).mean()
                    d_ref = pd.Series(ref_loss, index=pd.MultiIndex.from_arrays([per_model[ref_name][0]["origin"], per_model[ref_name][0]["asset"]])).groupby(level=0).mean()
                    _, p = dm_test(d_model.sort_index().to_numpy(), d_ref.sort_index().to_numpy())
                    row["dm_p"] = p
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "h", "pinball"]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, task: str, h: int) -> str:
    t = df[(df["task"] == task) & (df["h"] == h)].copy()
    point_name = "MAE(median)" if task == "logrv" else "R2_oos(median)"
    ref = REFERENCE[task]
    scale, unit = (1.0, "pinball") if task == "logrv" else (1000.0, "pinball x1000")
    lines = [
        f"| model | family | {unit} | skill vs {ref} | DM p | cov80 | calib | {point_name} | sec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        skill = "ref" if r["model"] == ref else f"{100 * r['skill_vs_ref']:+.1f}%"
        dm = "" if pd.isna(r.get("dm_p", np.nan)) else f"{r['dm_p']:.3f}"
        point = f"{r['point']:.3f}" if task == "logrv" else f"{100 * r['point']:+.2f}%"
        lines.append(
            f"| {r['model']} | {r['family']} | {scale * r['pinball']:.4f} | {skill} | {dm} | {r['cov80']:.3f} | "
            f"{r['calib']:.3f} | {point} | {r['seconds']:.0f} |"
        )
    return "\n".join(lines)


def make_figures(df: pd.DataFrame, scores: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "classical": "#7f7f7f",
        "foundation zero-shot": "#1f77b4",
        "foundation + LoRA": "#d62728",
        "trained from scratch": "#2ca02c",
    }
    panels = [("logrv", 1), ("logrv", 5), ("ret", 1)]

    # 1) skill vs reference
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (task, h) in zip(axes, panels):
        t = scores[(scores["task"] == task) & (scores["h"] == h)].sort_values("skill_vs_ref")
        ax.barh(t["model"], 100 * t["skill_vs_ref"], color=[colors[f] for f in t["family"]])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{TASK_LABEL[task]}, h = {h}\n(pinball skill vs {REFERENCE[task]}, %)", fontsize=10)
        ax.tick_params(labelsize=8)
    colors = {f: c for f, c in colors.items() if f in set(scores["family"])}
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, colors.keys(), loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(fig_dir / "skill.png", dpi=140)
    plt.close(fig)

    # 2) calibration
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (task, h) in zip(axes, panels):
        g = df[(df["task"] == task) & (df["h"] == h)]
        for model, gm in g.groupby("model"):
            ax.plot(LEVELS, calibration(gm["y"].to_numpy(), gm[QCOLS].to_numpy()), marker="o", ms=3, lw=1,
                    color=colors[family(model)], alpha=0.8)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_title(f"{TASK_LABEL[task]}, h = {h}\n(empirical frequency of y <= quantile)", fontsize=10)
        ax.set_xlabel("nominal level")
    fig.legend(handles, colors.keys(), loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(fig_dir / "calibration.png", dpi=140)
    plt.close(fig)

    # 3) cumulative loss difference vs reference through the test period (volatility, h = 1)
    task, h = "logrv", 1
    g = df[(df["task"] == task) & (df["h"] == h)]
    daily = {m: pd.Series(pinball(x["y"].to_numpy(), x[QCOLS].to_numpy()), index=pd.MultiIndex.from_arrays([x["origin"], x["asset"]])).groupby(level=0).mean().sort_index() for m, x in g.groupby("model")}
    ref = daily[REFERENCE[task]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for m, s in daily.items():
        if m == REFERENCE[task]:
            continue
        ax.plot((ref - s).cumsum(), label=m, color=colors[family(m)], lw=1.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Cumulative pinball gain over {REFERENCE[task]} - {TASK_LABEL[task]}, h = 1 (up = better)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "cumulative_gain.png", dpi=140)
    plt.close(fig)


def main() -> None:
    df = align(load_all())
    scores = score(df)
    tables = RESULTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    scores.to_csv(tables / "scores.csv", index=False)
    parts = []
    for task, h in [("logrv", 1), ("logrv", 5), ("ret", 1)]:
        if ((scores["task"] == task) & (scores["h"] == h)).any():
            n = int(scores[(scores["task"] == task) & (scores["h"] == h)]["n"].iloc[0])
            parts.append(f"### {TASK_LABEL[task]}, horizon {h} day{'s' if h > 1 else ''}  (n = {n} forecasts)\n\n" + markdown_table(scores, task, h))
    (RESULTS_DIR / "summary.md").write_text("\n\n".join(parts) + "\n")
    print("\n\n".join(parts))
    make_figures(df, scores)


if __name__ == "__main__":
    main()

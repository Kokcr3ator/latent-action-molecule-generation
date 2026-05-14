#!/usr/bin/env python3
"""
Generate paper figures from W&B run data.

Usage:
    python -m scripts.plot_figures [--out-dir figures/] [--samples 500]
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import wandb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolve default output dir relative to this file so it works regardless of
# the current working directory: code/scripts/plot_figures.py -> ../../paper/figures
_HERE = Path(__file__).resolve().parent          # code/scripts/
_DEFAULT_OUT_DIR = str(_HERE.parent.parent / "paper" / "figures")

WANDB_ENTITY = "latent-action-interdiff"
WANDB_PROJECT = "interdiff"

# Okabe-Ito colorblind-friendly palette (8 colours incl. black)
PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def fetch_runs(group: str, extra_filters: dict | None = None) -> list:
    api = wandb.Api(timeout=60)
    filters: dict = {"group": group, "state": "finished"}
    if extra_filters:
        filters.update(extra_filters)
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters)
    return list(runs)


def run_history(run, train_key: str, val_key: str, samples: int = 500) -> pd.DataFrame:
    """Return a DataFrame with columns [step, train, val]."""
    df = run.history(keys=["step", train_key, val_key], samples=samples)
    df = df.rename(columns={train_key: "train", val_key: "val"})
    # Use the logged "step" metric as x-axis; fall back to _step
    if "step" in df.columns and df["step"].notna().any():
        df = df.rename(columns={"step": "x"})
    else:
        df = df.rename(columns={"_step": "x"})
    return df[["x", "train", "val"]].copy()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _interpolate(df: pd.DataFrame, key: str, grid: np.ndarray) -> np.ndarray:
    sub = df[["x", key]].dropna()
    if len(sub) < 2:
        return np.full(len(grid), np.nan)
    return np.interp(grid, sub["x"].values, sub[key].values,
                     left=np.nan, right=np.nan)


def aggregate(
    dfs: list[pd.DataFrame],
    key: str,
    n_points: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate to a common step grid; return (steps, mean, p5, p95)."""
    valid = [df for df in dfs if not df.empty]
    if not valid:
        dummy = np.full(n_points, np.nan)
        return np.linspace(0, 1, n_points), dummy, dummy, dummy

    all_x = np.concatenate([df["x"].dropna().values for df in valid])
    grid = np.linspace(all_x.min(), all_x.max(), n_points)

    curves = np.stack([_interpolate(df, key, grid) for df in valid])  # (S, T)
    mean = np.nanmean(curves, axis=0)
    p5   = np.nanpercentile(curves,  5, axis=0)
    p95  = np.nanpercentile(curves, 95, axis=0)
    return grid, mean, p5, p95


# ---------------------------------------------------------------------------
# Log-scale translation
# ---------------------------------------------------------------------------

def compute_offset(group_dfs: dict[int, list[pd.DataFrame]]) -> float:
    """Global minimum across all train+val values (to ensure log > 0 after shift)."""
    vals = []
    for dfs in group_dfs.values():
        for df in dfs:
            for col in ("train", "val"):
                if col in df.columns:
                    vals.extend(df[col].dropna().tolist())
    if not vals:
        return 0.0
    return max(0.0, float(np.min(vals)) - 1e-6)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scaling(
    group_dfs: dict[int, list[pd.DataFrame]],
    sweep_label: str,
    train_label: str,
    val_label: str,
    title: str,
    out_stem: str,
) -> None:
    sweep_values = sorted(group_dfs.keys())
    offset = compute_offset(group_dfs)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, sv in enumerate(sweep_values):
        color = PALETTE[i % len(PALETTE)]
        label = f"{sweep_label} = {sv}"
        dfs = group_dfs[sv]

        for key, ls, suffix in [("train", "-", "train"), ("val", "--", "val")]:
            steps, mean, p5, p95 = aggregate(dfs, key)

            # Translate so that log is always defined, then apply log scale
            # via yscale='log': we shift the *data* so the minimum > 0.
            mean_t = mean - offset
            p5_t   = p5   - offset
            p95_t  = p95  - offset

            lbl = f"{label}" if suffix == "train" else "_nolegend_"
            ax.plot(steps, mean_t, color=color, linestyle=ls,
                    linewidth=1.5, label=lbl)
            ax.fill_between(steps, p5_t, p95_t,
                            color=color, alpha=0.12, linewidth=0)

    # Dummy lines for train/val legend entries
    ax.plot([], [], color="gray", linestyle="-",  linewidth=1.5, label="train")
    ax.plot([], [], color="gray", linestyle="--", linewidth=1.5, label="val")

    ax.set_yscale("log")
    ax.set_xlabel("Step", fontsize=12)
    offset_str = f" − {offset:.3g}" if offset > 0 else ""
    ax.set_ylabel(f"loss{offset_str}  (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc="upper right",
              framealpha=0.85, edgecolor="0.8")
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: GPT vocab scaling
# ---------------------------------------------------------------------------

def figure_gpt_vocab_scaling(out_dir: str, samples: int) -> None:
    print("Figure 1 — GPT vocab scaling (exp7)...")
    runs = fetch_runs("exp7_gpt_vocab_scaling")
    # Keep only base (small) GPT runs
    runs = [r for r in runs if re.match(r"pretrain_base_vocab\d+_seed\d+", r.name)]

    group_dfs: dict[int, list[pd.DataFrame]] = {}
    for run in runs:
        m = re.search(r"pretrain_base_vocab(\d+)_seed\d+", run.name)
        if not m:
            continue
        vocab_size = int(m.group(1))
        df = run_history(run, "train/loss", "val/loss", samples=samples)
        group_dfs.setdefault(vocab_size, []).append(df)

    if not group_dfs:
        print("  no runs found — skipping")
        return

    plot_scaling(
        group_dfs=group_dfs,
        sweep_label="vocab size",
        train_label="train/loss",
        val_label="val/loss",
        title="GPT pretraining: loss vs vocabulary size",
        out_stem=os.path.join(out_dir, "fig1_gpt_vocab_scaling"),
    )


# ---------------------------------------------------------------------------
# Figure 2: CGPT codebook scaling
# ---------------------------------------------------------------------------

def figure_cgpt_codebook_scaling(out_dir: str, samples: int) -> None:
    print("Figure 2 — CGPT codebook scaling (exp3)...")
    runs = fetch_runs("exp3_lam_scaling")
    # Keep only pretrain runs (not RL/distillation)
    runs = [r for r in runs
            if re.match(r"pretrain_controllable_vocab\d+_nlatent\d+_seed\d+", r.name)]

    group_dfs: dict[int, list[pd.DataFrame]] = {}
    for run in runs:
        m = re.search(r"nlatent(\d+)_seed\d+", run.name)
        if not m:
            continue
        num_latents = int(m.group(1))
        df = run_history(run, "train/loss", "val/loss", samples=samples)
        group_dfs.setdefault(num_latents, []).append(df)

    if not group_dfs:
        print("  no runs found — skipping")
        return

    plot_scaling(
        group_dfs=group_dfs,
        sweep_label="num latents",
        train_label="train/loss",
        val_label="val/loss",
        title="ControllableGPT pretraining: loss vs codebook size",
        out_stem=os.path.join(out_dir, "fig2_cgpt_codebook_scaling"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from W&B data")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR,
                        help="Directory to write figures into")
    parser.add_argument("--samples", type=int, default=500,
                        help="W&B history sample points per run (default: 500)")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    figure_gpt_vocab_scaling(args.out_dir, args.samples)
    figure_cgpt_codebook_scaling(args.out_dir, args.samples)

    print("Done.")


if __name__ == "__main__":
    main()

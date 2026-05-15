#!/usr/bin/env python3
"""
Generate paper figures from W&B run data.

Usage:
    python -m scripts.plot_figures [--out-dir figures/] [--samples 500] [--force-fetch]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import wandb

# ---------------------------------------------------------------------------
# Font setup — TeX Gyre Termes (free Times New Roman clone, NeurIPS/TMLR style)
# Bundled OTF files live next to the output figures so the script is OS-agnostic.
# ---------------------------------------------------------------------------
_FONTS_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures" / "fonts"
if _FONTS_DIR.is_dir():
    for _otf in _FONTS_DIR.glob("*.otf"):
        fm.fontManager.addfont(str(_otf))

# Pick the best available Times-compatible family
def _best_serif() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("TeX Gyre Termes", "Times New Roman", "Times", "serif"):
        if candidate in available or candidate == "serif":
            return candidate
    return "serif"

matplotlib.rcParams.update({
    "font.family": _best_serif(),
    "font.size": 7,
})

# 0.9 × textwidth for a standard article (letter paper, 1-in margins → 6.5 in)
_FIG_W = 0.9 * 6.5          # 5.85 in
_FIG_H = _FIG_W * 2 / 3    # 3:2 ratio ≈ 3.90 in

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolve default output dir relative to this file so it works regardless of
# the current working directory: code/scripts/plot_figures.py -> ../../paper/figures
_HERE = Path(__file__).resolve().parent          # code/scripts/
_DEFAULT_OUT_DIR = str(_HERE.parent.parent / "paper" / "figures")

WANDB_ENTITY = "latent-action-interdiff"
WANDB_PROJECT = "interdiff"

_RED = "#FF0000"  # red for scaling-curve figures

# Paul Tol's Muted qualitative palette — colorblind-safe, 9 colours
PALETTE = [
    "#CC6677",  # rose
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#117733",  # green
    "#88CCEE",  # cyan
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # purple
]

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(out_dir: str, stem: str) -> str:
    return os.path.join(out_dir, f"{stem}_data.json")


def _save_cache(path: str, group_dfs: dict[int, list[pd.DataFrame]]) -> None:
    data = {
        str(sv): [
            {"x": df["x"].tolist(), "train": df["train"].tolist(), "val": df["val"].tolist()}
            for df in dfs
        ]
        for sv, dfs in group_dfs.items()
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  cached source data → {path}")


def _load_cache(path: str) -> dict[int, list[pd.DataFrame]]:
    with open(path) as f:
        data = json.load(f)
    group_dfs: dict[int, list[pd.DataFrame]] = {
        int(sv_str): [pd.DataFrame(r) for r in records]
        for sv_str, records in data.items()
    }
    print(f"  loaded source data from cache {path}")
    return group_dfs


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
    """Return a DataFrame with columns [x, train, val]."""
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
# Plotting
# ---------------------------------------------------------------------------

def plot_scaling(
    group_dfs: dict[int, list[pd.DataFrame]],
    sweep_label: str,
    title: str,
    out_stem: str,
) -> None:
    sweep_values = sorted(group_dfs.keys())

    # Pass 1: compute all aggregates
    all_stats: dict = {}
    for sv in sweep_values:
        all_stats[sv] = {
            key: aggregate(group_dfs[sv], key)
            for key in ("train", "val")
        }

    # Translation constant: smallest c >= 0 such that min(p5) + c > 0, plus a tiny
    # epsilon so the translated minimum sits comfortably above zero on the log axis.
    all_p5 = [all_stats[sv][k][2] for sv in sweep_values for k in ("train", "val")]
    global_min = float(np.nanmin(np.concatenate([v[np.isfinite(v)] for v in all_p5])))
    c = max(0.0, -global_min) + 1e-6
    if c > 0:
        print(f"  y-translation c = {c:.3e}  (loss + c plotted on log axis)")

    # Derive y limits from translated mean curves
    all_means = [all_stats[sv][k][1] for sv in sweep_values for k in ("train", "val")]
    translated = np.concatenate([(m + c)[np.isfinite(m + c) & ((m + c) > 0)]
                                 for m in all_means])
    ymin = float(np.nanmin(translated)) * 0.5 if translated.size else 1e-4
    ymax = float(np.nanmax(translated)) * 2.0 if translated.size else 10.0

    # Pass 2: plot translated values
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    for i, sv in enumerate(sweep_values):
        color = PALETTE[i % len(PALETTE)]
        for key, ls in [("train", "-"), ("val", "--")]:
            steps, mean, p5, p95 = all_stats[sv][key]
            lbl = f"{sweep_label}={sv}" if ls == "-" else "_nolegend_"
            ax.plot(steps, mean + c, color=color, linestyle=ls,
                    linewidth=0.5, label=lbl)
            ax.fill_between(steps, p5 + c, p95 + c,
                            color=color, alpha=0.12, linewidth=0)

    # set_yscale must come before set_ylim or autoscale will override the limits
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ylabel = f"loss + {c:.0e}" if c > 0 else "loss"
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5), numticks=15))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(0, 20_000)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5000))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: "0" if x == 0 else f"{int(x / 1000)}K"
    ))

    # Legend below the axes — max 4 cols so it wraps rather than overflowing the width
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=min(len(sweep_values), 5),
        fontsize=7,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.8,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("0.6")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)   # room for 2-row external legend
    path = f"{out_stem}.png"
    fig.savefig(path, dpi=300)
    print(f"  saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined scaling-curve plot (min val loss vs size, two series)
# ---------------------------------------------------------------------------

def _min_val_stats(group_dfs: dict[int, list[pd.DataFrame]]):
    """Return (xs, means, p5s, p95s) arrays of min-val-loss per sweep value."""
    xs, means, p5s, p95s = [], [], [], []
    for sv in sorted(group_dfs.keys()):
        mins = [float(df["val"].dropna().min()) for df in group_dfs[sv]
                if df["val"].dropna().size > 0]
        if not mins:
            continue
        arr = np.array(mins)
        xs.append(sv)
        means.append(float(np.mean(arr)))
        p5s.append(float(np.percentile(arr,  5)))
        p95s.append(float(np.percentile(arr, 95)))
    return (np.array(xs, dtype=float), np.array(means), np.array(p5s), np.array(p95s))


def plot_combined_scaling_curve(
    series: list[tuple],   # [(xs, means, p5s, p95s, color, label), ...]
    title: str,
    out_stem: str,
) -> None:
    # Shared translation: ensure all p5 values are positive on the log axis
    global_min = min(float(np.min(p5s)) for _, _, p5s, _, _, _ in series)
    c = max(0.0, -global_min) + 1e-6
    if c > 1e-5:
        print(f"  y-translation c = {c:.3e}")

    # y limits from the combined translated means
    all_tm = np.concatenate([means + c for _, means, _, _, _, _ in series])
    ymin = max(float(np.min(all_tm)) * 0.5, 1e-4)
    ymax = float(np.max(all_tm)) * 2.0

    # union of all x values for tick placement
    all_xs = sorted({int(v) for _, means, _, _, _, _ in series
                     for xs, *_ in [series[0]] for v in xs}
                    | {int(v) for xs, *_ in series for v in xs})

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    for xs, means, p5s, p95s, color, label in series:
        tm = means + c
        yerr_lo = tm - np.maximum(p5s + c, ymin)
        yerr_hi = np.minimum(p95s + c, ymax) - tm
        ax.errorbar(xs, tm, yerr=[yerr_lo, yerr_hi],
                    color=color, linewidth=0.7,
                    marker="o", markersize=3,
                    elinewidth=0.5, capsize=2, capthick=0.5,
                    label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(sorted(all_xs))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v/1000)}K" if v >= 1000 else str(int(v))
    ))
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5), numticks=15))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    ylabel = f"min val loss + {c:.0e}" if c > 1e-5 else "min val loss"
    ax.set_xlabel("size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.legend(fontsize=7, frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("0.6")

    fig.tight_layout()
    path = f"{out_stem}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: GPT vocab scaling
# ---------------------------------------------------------------------------

def figure_gpt_vocab_scaling(out_dir: str, samples: int, force_fetch: bool) -> None:
    print("Figure 1 — GPT vocab scaling (exp7)...")
    cache = _cache_path(out_dir, "fig1_gpt_vocab_scaling")

    if not force_fetch and os.path.exists(cache):
        group_dfs = _load_cache(cache)
    else:
        runs = fetch_runs("exp7_gpt_vocab_scaling")
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

        _save_cache(cache, group_dfs)

    plot_scaling(
        group_dfs=group_dfs,
        sweep_label="vocab size",
        title="Scaling vocabulary size",
        out_stem=os.path.join(out_dir, "fig1_gpt_vocab_scaling"),
    )


# ---------------------------------------------------------------------------
# Figure 2: CGPT codebook scaling
# ---------------------------------------------------------------------------

def figure_cgpt_codebook_scaling(out_dir: str, samples: int, force_fetch: bool) -> None:
    print("Figure 2 — CGPT codebook scaling (exp3)...")
    cache = _cache_path(out_dir, "fig2_cgpt_codebook_scaling")

    if not force_fetch and os.path.exists(cache):
        group_dfs = _load_cache(cache)
    else:
        runs = fetch_runs("exp3_lam_scaling")
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

        _save_cache(cache, group_dfs)

    plot_scaling(
        group_dfs=group_dfs,
        sweep_label="num latents",
        title="Scaling codebook size",
        out_stem=os.path.join(out_dir, "fig2_cgpt_codebook_scaling"),
    )


# ---------------------------------------------------------------------------
# Figure 3: Combined scaling curve (vocab size in black, codebook size in red)
# ---------------------------------------------------------------------------

def figure_combined_scaling(out_dir: str, samples: int, force_fetch: bool) -> None:
    print("Figure 3 — Combined scaling curve (exp7 + exp3)...")

    # GPT data — reuse fig1 cache
    cache1 = _cache_path(out_dir, "fig1_gpt_vocab_scaling")
    if not force_fetch and os.path.exists(cache1):
        gpt_dfs = _load_cache(cache1)
    else:
        runs = fetch_runs("exp7_gpt_vocab_scaling")
        runs = [r for r in runs if re.match(r"pretrain_base_vocab\d+_seed\d+", r.name)]
        gpt_dfs: dict[int, list[pd.DataFrame]] = {}
        for run in runs:
            m = re.search(r"pretrain_base_vocab(\d+)_seed\d+", run.name)
            if not m:
                continue
            gpt_dfs.setdefault(int(m.group(1)), []).append(
                run_history(run, "train/loss", "val/loss", samples=samples))
        if not gpt_dfs:
            print("  no GPT runs found — skipping")
            return
        _save_cache(cache1, gpt_dfs)

    # CGPT data — reuse fig2 cache
    cache2 = _cache_path(out_dir, "fig2_cgpt_codebook_scaling")
    if not force_fetch and os.path.exists(cache2):
        cgpt_dfs = _load_cache(cache2)
    else:
        runs = fetch_runs("exp3_lam_scaling")
        runs = [r for r in runs
                if re.match(r"pretrain_controllable_vocab\d+_nlatent\d+_seed\d+", r.name)]
        cgpt_dfs: dict[int, list[pd.DataFrame]] = {}
        for run in runs:
            m = re.search(r"nlatent(\d+)_seed\d+", run.name)
            if not m:
                continue
            cgpt_dfs.setdefault(int(m.group(1)), []).append(
                run_history(run, "train/loss", "val/loss", samples=samples))
        if not cgpt_dfs:
            print("  no CGPT runs found — skipping")
            return
        _save_cache(cache2, cgpt_dfs)

    series = [
        (*_min_val_stats(gpt_dfs),  PALETTE[0], "vocab size"),
        (*_min_val_stats(cgpt_dfs), PALETTE[1], "codebook size"),
    ]
    plot_combined_scaling_curve(
        series=series,
        title="Scaling: vocabulary size vs codebook size",
        out_stem=os.path.join(out_dir, "fig3_combined_scaling_curve"),
    )


# ---------------------------------------------------------------------------
# Figure 4: Policy distillation vs GPT — val loss comparison
# ---------------------------------------------------------------------------

def figure_distillation_vs_gpt(out_dir: str, samples: int, force_fetch: bool) -> None:
    print("Figure 4 — Policy distillation vs GPT val loss (policydistillation + exp7)...")

    # -- GPT runs (reuse fig1 cache) --
    cache_gpt = _cache_path(out_dir, "fig1_gpt_vocab_scaling")
    if not force_fetch and os.path.exists(cache_gpt):
        gpt_dfs = _load_cache(cache_gpt)
    else:
        runs = fetch_runs("exp7_gpt_vocab_scaling")
        runs = [r for r in runs if re.match(r"pretrain_base_vocab\d+_seed\d+", r.name)]
        gpt_dfs: dict[int, list[pd.DataFrame]] = {}
        for run in runs:
            m = re.search(r"pretrain_base_vocab(\d+)_seed\d+", run.name)
            if not m:
                continue
            gpt_dfs.setdefault(int(m.group(1)), []).append(
                run_history(run, "train/loss", "val/loss", samples=samples))
        if not gpt_dfs:
            print("  no GPT runs found — skipping")
            return
        _save_cache(cache_gpt, gpt_dfs)

    # -- Policy distillation runs --
    cache_dist = _cache_path(out_dir, "fig4_policy_distillation")
    if not force_fetch and os.path.exists(cache_dist):
        dist_dfs = _load_cache(cache_dist)
    else:
        runs = fetch_runs("exp6_distillation_impact")
        runs = [r for r in runs if re.match(r"policydistillation_nlatents\d+_vocab\d+_seed\d+", r.name)]
        dist_dfs: dict[int, list[pd.DataFrame]] = {}
        for run in runs:
            m = re.search(r"nlatents(\d+)_vocab\d+_seed\d+", run.name)
            if not m:
                continue
            dist_dfs.setdefault(int(m.group(1)), []).append(
                run_history(run, "train/loss", "val/loss", samples=samples))
        if not dist_dfs:
            print("  no policy distillation runs found — skipping")
            return
        _save_cache(cache_dist, dist_dfs)

    series = [
        (*_min_val_stats(gpt_dfs),  PALETTE[0], "GPT (vocab size)"),
        (*_min_val_stats(dist_dfs), PALETTE[1], "Policy distillation (codebook size)"),
    ]
    plot_combined_scaling_curve(
        series=series,
        title="Policy distillation vs GPT: min val loss vs size",
        out_stem=os.path.join(out_dir, "fig4_distillation_vs_gpt"),
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
    parser.add_argument("--force-fetch", action="store_true",
                        help="Re-fetch from W&B even when a local cache exists")
    parser.add_argument("--figure", type=int, default=None,
                        help="Run only this figure number (1–4); omit to run all")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    figs = {
        1: lambda: figure_gpt_vocab_scaling(args.out_dir, args.samples, args.force_fetch),
        2: lambda: figure_cgpt_codebook_scaling(args.out_dir, args.samples, args.force_fetch),
        3: lambda: figure_combined_scaling(args.out_dir, args.samples, args.force_fetch),
        4: lambda: figure_distillation_vs_gpt(args.out_dir, args.samples, args.force_fetch),
    }
    to_run = [args.figure] if args.figure else sorted(figs)
    for n in to_run:
        figs[n]()

    print("Done.")


if __name__ == "__main__":
    main()

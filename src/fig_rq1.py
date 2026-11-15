"""Figure for RQ1: how much the model beats each fund's own baseline.

Four small panels, one per sample (hand-picked hedge funds, Form ADV hedge
funds, long-only managers, N-PORT mutual funds). In each panel:
  x = the fund's majority-class baseline accuracy (%),
  y = LightGBM accuracy minus that baseline (percentage points),
one coloured dot per fund (size: number of test positions), all other samples'
funds in light grey behind for context, a zero line (no gain), and a black
diamond-with-ring at the sample's median fund. The mutual-fund average of
Cohen, Lu & Nguyen (52% -> 71%, a gain of 19 points) is a star.

Plotting the gain instead of model-vs-baseline turns the diagonal into a flat
line, so small differences are visible and no panel area is wasted.

Output: paper/figures/fig_rq1_funds.pdf (+ experiments/fig_rq1_funds.png)
Usage:  .venv/bin/python src/fig_rq1.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = [  # (folder, panel title, colour): categorical slots 1-4 in fixed order
    ("experiments", "Hedge funds, hand-picked", "#2a78d6"),
    ("experiments_adv", "Hedge funds, Form ADV", "#eb6834"),
    ("experiments_longonly", "Long-only managers", "#1baf7a"),
    ("experiments_nport", "Mutual funds, N-PORT", "#c98a00"),
]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.5, "axes.edgecolor": MUTED,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "pdf.fonttype": 42,
    })
    data = {}
    for folder, title, color in SAMPLES:
        d = pd.read_csv(ROOT / folder / "rq1_per_fund.csv")
        d["x"] = d["majority"] * 100
        d["g"] = (d["lightgbm"] - d["majority"]) * 100
        data[folder] = d
    allrows = pd.concat(data.values())
    smax = allrows["rows"].max()

    def size(rows):
        return 3 + 22 * np.sqrt(rows / smax)

    fig, axes = plt.subplots(2, 2, figsize=(3.33, 2.95), sharex=True, sharey=True)
    for ax, (folder, title, color) in zip(axes.flat, SAMPLES):
        d = data[folder]
        others = pd.concat([v for k, v in data.items() if k != folder])
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.scatter(others["x"], others["g"], s=size(others["rows"]) * 0.6, c="#d4d4d4",
                   linewidths=0, zorder=2)
        ax.scatter(d["x"], d["g"], s=size(d["rows"]), c=color, alpha=0.8,
                   edgecolors="white", linewidths=0.3, zorder=3)
        mx, mg = d["x"].median(), d["g"].median()
        ax.scatter([mx], [mg], s=34, marker="o", facecolors="white", edgecolors=INK,
                   linewidths=1.0, zorder=5)
        ax.scatter([mx], [mg], s=6, marker="o", c=INK, zorder=6)
        ax.scatter([52], [19], s=38, marker="*", c=INK, linewidths=0, zorder=5)
        ax.text(0.97, 0.97, f"{title}\n$n$ = {len(d)}", transform=ax.transAxes, ha="right",
                va="top", fontsize=6.5, color=INK, linespacing=1.15)
        ax.grid(color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[0, 0].set_xlim(10, 100)
    axes[0, 0].set_ylim(-30, 47)
    axes[0, 0].set_xticks([20, 40, 60, 80, 100])
    axes[0, 0].set_yticks([-20, 0, 20, 40])
    fig.supxlabel("Baseline accuracy: fund's most common action (%)", fontsize=7, y=0.015)
    fig.supylabel("LightGBM gain over baseline (points)", fontsize=7, x=0.015)
    fig.tight_layout(pad=0.25, h_pad=0.5, w_pad=0.5)
    fig.savefig(ROOT / "paper" / "figures" / "fig_rq1_funds.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(ROOT / "experiments" / "fig_rq1_funds.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved paper/figures/fig_rq1_funds.pdf")


if __name__ == "__main__":
    main()

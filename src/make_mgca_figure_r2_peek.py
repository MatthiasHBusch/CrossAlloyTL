"""
One-off R^2 version of the MgCa contrast figure for visual inspection.
NOT referenced from the manuscript --- the published figure stays in Pearson r
because R^2 for MgCa is so deeply negative that the values do not fit on a
shared axis with the in-cluster panel.

Two output variants:
  fig_mgca_contrast_r2.png          --- y in (-15, 1.0) (kNN-Tan goes off)
  fig_mgca_contrast_r2_symlog.png   --- symlog y (compresses extremes)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figures import (
    MODEL_ORDER, MODEL_COLORS, MODEL_LABELS, SETTINGS_ORDER, SETTING_LABELS,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = pd.read_csv(ROOT / "results" / "experiment_results.csv").dropna(subset=["r2"])
MGCA = pd.read_csv(ROOT / "results" / "mgca_contrast.csv").dropna(subset=["r2"])

def make(symlog: bool, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=not symlog,
                              gridspec_kw={"wspace": 0.12})

    for ax, (label, df) in zip(axes, [
        ("In-cluster (AZ31, AZ91, WE43)", MAIN),
        ("MgCa (outside cluster)", MGCA),
    ]):
        for model in MODEL_ORDER:
            sub = df[df["model"] == model]
            if len(sub) == 0:
                continue
            agg = sub.groupby("setting")["r2"].agg(["mean", "std", "count"])
            agg = agg.reindex(SETTINGS_ORDER)
            x = np.arange(len(SETTINGS_ORDER))
            ax.errorbar(x, agg["mean"], yerr=agg["std"] / np.sqrt(agg["count"]),
                        marker="o", markersize=5, linewidth=1.5,
                        color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                        capsize=2.5, alpha=0.85)
        ax.set_xticks(np.arange(len(SETTINGS_ORDER)))
        ax.set_xticklabels([SETTING_LABELS[s] for s in SETTINGS_ORDER], fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel(label, fontweight="bold")

    if symlog:
        for ax in axes:
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(-100, 1)
        axes[0].set_ylabel(r"$R^{2}$ vs. exp. IE (symlog)")
    else:
        axes[0].set_ylim(-1.5, 1.0)
        axes[1].set_ylim(-15, 1.0)
        axes[0].set_ylabel(r"$R^{2}$ vs. exp. IE (in-cluster)")
        axes[1].set_ylabel(r"$R^{2}$ vs. exp. IE (MgCa)")

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=9, framealpha=0.95)
    plt.tight_layout()
    out = ROOT / "figures" / fname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

make(symlog=False, fname="fig_mgca_contrast_r2.png")
make(symlog=True,  fname="fig_mgca_contrast_r2_symlog.png")

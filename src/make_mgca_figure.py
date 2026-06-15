"""MgCa contrast figure: in-cluster target alloys vs MgCa, all six models.

Polished line-plot style with a top legend (3 cols x 2 rows), matching
fig1_main. Compact figure size so the (absolute) fonts read larger.
Output: figures/fig_mgca_contrast.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import (SETTINGS_ORDER, SETTING_LABELS, MODEL_ORDER,
                          MODEL_COLORS, MODEL_LABELS, POLISHED_RC)

plt.rcParams.update(POLISHED_RC)

ROOT = Path(__file__).resolve().parents[1]
MAIN = pd.read_csv(ROOT / "results" / "experiment_results.csv").dropna(subset=["r2"])
MGCA = pd.read_csv(ROOT / "results" / "mgca_contrast.csv").dropna(subset=["r2"])

# 2-panel line plot, 15% less wide / 25% less high than the prior (9.6, 3.7).
fig, axes = plt.subplots(1, 2, figsize=(8.16, 2.78), gridspec_kw={"wspace": 0.30})
x = np.arange(len(SETTINGS_ORDER))
panels = [("In-cluster (AZ31, AZ91, WE43)", MAIN, (-1.5, 1.0)),
          ("MgCa (outside cluster)", MGCA, (-15.0, 1.0))]
for ax, (label, d, ylim) in zip(axes, panels):
    for m in MODEL_ORDER:
        s = d[d.model == m]
        if not len(s):
            continue
        a = s.groupby("setting")["r2"].agg(["mean", "std", "count"]).reindex(SETTINGS_ORDER)
        gem = (m == "Gemini3.1Pro")
        ax.errorbar(x, a["mean"], yerr=a["std"] / np.sqrt(a["count"]), marker="o",
                    ms=5 if gem else 4, lw=2.4 if gem else 1.3, alpha=1 if gem else 0.78,
                    color=MODEL_COLORS[m], label=MODEL_LABELS[m], capsize=2,
                    zorder=6 if gem else 3)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS[s] for s in SETTINGS_ORDER], fontsize=9)
    ax.set_xlabel(label, fontweight="bold")
    ax.grid(True, axis="y")
axes[0].set_ylabel(r"$R^{2}$ (in-cluster)")
axes[1].set_ylabel(r"$R^{2}$ (MgCa)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3,
           frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig_mgca_contrast.png", dpi=300, bbox_inches="tight")
fig.savefig(ROOT / "figures" / "fig_mgca_contrast.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig_mgca_contrast.png/pdf")

"""
Alloy SAR-similarity heatmap (main-text Fig. 3): pairwise Spearman r of
inhibitor rankings across the 11 Mg systems. Target alloys (AZ31, AZ91,
WE43) are outlined; group separators delimit commercial / high-Fe pure /
Ca-grain-boundary blocks.
Output: figures/fig5_alloy_correlation.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from make_figures import POLISHED_RC
plt.rcParams.update(POLISHED_RC)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_OUT = ROOT / "tl_crossalloy" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

TARGET = {"AZ31", "AZ91", "WE43"}
SHORT = {"CPMg220ppmFe": "CPMg220", "CPMg342ppmFe": "CPMg342",
         "HPMg50ppmFe": "HPMg50", "HPMg51ppmFe": "HPMg51"}
sh = lambda a: SHORT.get(a, a)
GROUP_ORDER = ["AM50", "AZ31", "AZ91", "E21", "HPMg51ppmFe", "WE43", "ZE41",
               "CPMg220ppmFe", "CPMg342ppmFe", "HPMg50ppmFe", "MgCa"]
NC, NHF = 7, 3  # commercial / high-Fe block sizes (separators after these)

spm = pd.read_csv(RESULTS_DIR / "spearman_matrix.csv", index_col=0)
spm = spm.loc[GROUP_ORDER, GROUP_ORDER].values
n = len(GROUP_ORDER)
labels = [sh(a) for a in GROUP_ORDER]

fig, ax = plt.subplots(figsize=(5.6, 5.0))
im = ax.imshow(spm, cmap=plt.cm.Blues, vmin=0, vmax=1, aspect="equal")
for i in range(n):
    for j in range(n):
        v = spm[i, j]
        txt = "—" if i == j else f"{v:.2f}"
        col = "white" if (i == j or v > 0.6) else "0.25"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=col)

# Group separators
for k in [NC - 0.5, NC + NHF - 0.5]:
    ax.axhline(k, color="0.35", lw=1.4, ls="--")
    ax.axvline(k, color="0.35", lw=1.4, ls="--")
# Target-alloy outlines
for idx, a in enumerate(GROUP_ORDER):
    if a in TARGET:
        ax.add_patch(mpatches.Rectangle((idx - 0.5, -0.5), 1, n, lw=1.8,
                     edgecolor="#C0392B", facecolor="none", clip_on=False))
        ax.add_patch(mpatches.Rectangle((-0.5, idx - 0.5), n, 1, lw=1.8,
                     edgecolor="#C0392B", facecolor="none", clip_on=False))

ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
fig.colorbar(im, ax=ax, label=r"Spearman $r$", fraction=0.045, pad=0.03)
fig.tight_layout()
fig.savefig(FIG_OUT / "fig5_alloy_correlation.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_OUT / "fig5_alloy_correlation.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved fig5_alloy_correlation.png/pdf")

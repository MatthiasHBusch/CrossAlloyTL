"""
SI Fig. S4: two representative alloy pairs in IE-space:
  - ZE41 vs MgCa        (cluster vs outside-cluster contrast)
  - HPMg50 vs HPMg51    (high-Fe pair, weak correlation)
Symlog axes spread the strong-accelerator tail so the structure is visible.
Spearman r and n are computed on the full data.
Output: figures/fig_si_alloy_pair_scatters.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from make_figures import POLISHED_RC
plt.rcParams.update(POLISHED_RC)

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "ExCorrDatasetClean.csv"
FIG_OUT = ROOT / "tl_crossalloy" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

PAIRS = [("ZE41", "MgCa", "#C0392B"), ("HPMg50ppmFe", "HPMg51ppmFe", "#E67E22")]
SHORT = {"HPMg50ppmFe": "HPMg50", "HPMg51ppmFe": "HPMg51"}
sh = lambda a: SHORT.get(a, a)
VMIN, VMAX, LINTHRESH = -500, 100, 20

df = pd.read_csv(DATA_CSV)
mg = df[df["BaseMaterial"] == "Mg"].copy()
agg = mg.groupby(["Alloy", "canonical_SMILES"])["IE"].mean().reset_index()


def common_ie(a, b):
    da = agg[agg.Alloy == a].set_index("canonical_SMILES")["IE"]
    db = agg[agg.Alloy == b].set_index("canonical_SMILES")["IE"]
    c = da.index.intersection(db.index)
    return da.loc[c].values, db.loc[c].values


fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7))
for ax, (a, b, color) in zip(axes, PAIRS):
    x, y = common_ie(a, b)
    r, _ = spearmanr(x, y)
    ax.scatter(x, y, s=26, color=color, alpha=0.6, edgecolor="white", linewidth=0.3)
    ax.plot([VMIN, VMAX], [VMIN, VMAX], "--", color="0.4", lw=1, label="y = x")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_xlim(VMIN, VMAX); ax.set_ylim(VMIN, VMAX)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(f"IE on {sh(a)} (%)")
    ax.set_ylabel(f"IE on {sh(b)} (%)")
    ax.text(0.04, 0.96, f"Spearman r = {r:+.2f}\nn = {len(x)}", ha="left", va="top",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, pad=3))
    ax.legend(loc="lower right", fontsize=9)
    print(f"  {sh(a):8s} vs {sh(b):8s}: Spearman r = {r:+.3f}, n = {len(x)}")

fig.suptitle("Representative alloy pairs in IE-space (symlog axes)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIG_OUT / "fig_si_alloy_pair_scatters.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_OUT / "fig_si_alloy_pair_scatters.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_OUT/'fig_si_alloy_pair_scatters.png'}")

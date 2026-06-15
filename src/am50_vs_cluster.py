"""
SI Fig. S3: AM50 IE against each of the six other commercial-cluster alloys.
Symlog axes (linear near 0, log for the strong-accelerator tail) so the
near-diagonal cluster structure is visible rather than crushed into the
top-right corner. Spearman r and n are computed on the full data.
Output: figures/fig_am50_vs_cluster.{png,pdf}
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

X_ALLOY = "AM50"
OTHERS = ["AZ31", "AZ91", "E21", "HPMg51ppmFe", "WE43", "ZE41"]
SHORT = {"HPMg51ppmFe": "HPMg51"}
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


fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0))
for ax, other in zip(axes.flat, OTHERS):
    x, y = common_ie(X_ALLOY, other)
    r, _ = spearmanr(x, y)
    ax.scatter(x, y, s=26, color="#2471A3", alpha=0.6, edgecolor="white", linewidth=0.3)
    ax.plot([VMIN, VMAX], [VMIN, VMAX], "--", color="0.4", lw=1, label="y = x")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_xlim(VMIN, VMAX); ax.set_ylim(VMIN, VMAX)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(f"IE on {X_ALLOY} (%)")
    ax.set_ylabel(f"IE on {sh(other)} (%)")
    ax.text(0.04, 0.96, f"Spearman r = {r:+.2f}\nn = {len(x)}", ha="left", va="top",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, pad=3))
    ax.legend(loc="lower right", fontsize=9)
    print(f"  {X_ALLOY} vs {sh(other):8s}: Spearman r = {r:+.3f}, n = {len(x)}")

fig.suptitle("AM50 vs. each commercial-cluster alloy (symlog axes)", fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(FIG_OUT / "fig_am50_vs_cluster.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_OUT / "fig_am50_vs_cluster.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {FIG_OUT/'fig_am50_vs_cluster.png'}")

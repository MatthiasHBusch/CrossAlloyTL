"""
Pearson counterpart to make_fig1_heatmap.py.
Reads results/pearson_matrix.csv (dir 11) and writes
tl_crossalloy/figures/fig_alloy_correlation_pearson.{png,pdf}.
Not referenced from the manuscript --- kept as a supplementary/inspection figure.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_OUT = ROOT / "tl_crossalloy" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

TARGET_ALLOYS = {"AZ31", "AZ91", "WE43"}

SHORT = {
    "AM50": "AM50", "AZ31": "AZ31", "AZ91": "AZ91",
    "CPMg220ppmFe": "CPMg220", "CPMg342ppmFe": "CPMg342",
    "E21": "E21", "HPMg50ppmFe": "HPMg50", "HPMg51ppmFe": "HPMg51",
    "MgCa": "MgCa", "WE43": "WE43", "ZE41": "ZE41",
}
GROUPS = {
    "AM50": "Commercial", "AZ31": "Commercial", "AZ91": "Commercial",
    "E21": "Commercial", "HPMg51ppmFe": "Commercial",
    "WE43": "Commercial", "ZE41": "Commercial",
    "CPMg220ppmFe": "High-Fe pure", "CPMg342ppmFe": "High-Fe pure",
    "HPMg50ppmFe": "High-Fe pure",
    "MgCa": "Ca grain-bdry",
}
GROUP_COLORS = {
    "Commercial":     "#2471A3",
    "High-Fe pure":   "#E67E22",
    "Ca grain-bdry":  "#C0392B",
}
GROUP_ORDER = [
    "AM50", "AZ31", "AZ91", "E21", "HPMg51ppmFe", "WE43", "ZE41",
    "CPMg220ppmFe", "CPMg342ppmFe", "HPMg50ppmFe", "MgCa",
]

pearson = pd.read_csv(RESULTS_DIR / "pearson_matrix.csv", index_col=0)

fig, ax_heat = plt.subplots(figsize=(5.2, 4.8))

pm = pearson.loc[GROUP_ORDER, GROUP_ORDER]
display_labels = [SHORT[a] for a in GROUP_ORDER]
cmap = plt.cm.Blues
im = ax_heat.imshow(pm.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

for i in range(len(GROUP_ORDER)):
    for j in range(len(GROUP_ORDER)):
        val = pm.values[i, j]
        if i == j:
            ax_heat.text(j, i, "—", ha="center", va="center",
                         fontsize=9, color="white")
        else:
            color = "white" if val > 0.65 else "black"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=8.5, color=color)

for idx, alloy in enumerate(GROUP_ORDER):
    if alloy in TARGET_ALLOYS:
        rect = mpatches.Rectangle((idx - 0.5, -0.5), 1, len(GROUP_ORDER),
                                  linewidth=2, edgecolor="black",
                                  facecolor="none", clip_on=False)
        ax_heat.add_patch(rect)
        rect2 = mpatches.Rectangle((-0.5, idx - 0.5), len(GROUP_ORDER), 1,
                                   linewidth=2, edgecolor="black",
                                   facecolor="none", clip_on=False)
        ax_heat.add_patch(rect2)

ax_heat.set_xticks(range(len(GROUP_ORDER)))
ax_heat.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
ax_heat.set_yticks(range(len(GROUP_ORDER)))
ax_heat.set_yticklabels(display_labels, fontsize=10)
for tick, alloy in zip(ax_heat.get_xticklabels(), GROUP_ORDER):
    tick.set_color(GROUP_COLORS[GROUPS[alloy]])
    if alloy in TARGET_ALLOYS:
        tick.set_fontweight("bold")
for tick, alloy in zip(ax_heat.get_yticklabels(), GROUP_ORDER):
    tick.set_color(GROUP_COLORS[GROUPS[alloy]])
    if alloy in TARGET_ALLOYS:
        tick.set_fontweight("bold")

n_commercial = sum(1 for a in GROUP_ORDER if GROUPS[a] == "Commercial")
n_high_fe = sum(1 for a in GROUP_ORDER if GROUPS[a] == "High-Fe pure")
for n in [n_commercial - 0.5, n_commercial + n_high_fe - 0.5]:
    ax_heat.axhline(n, color="gray", lw=1.5, ls="--")
    ax_heat.axvline(n, color="gray", lw=1.5, ls="--")

plt.colorbar(im, ax=ax_heat, label=r"Pearson $r$",
             fraction=0.045, pad=0.02)
plt.tight_layout()
fig.savefig(FIG_OUT / "fig_alloy_correlation_pearson.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_OUT / "fig_alloy_correlation_pearson.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_OUT/'fig_alloy_correlation_pearson.png'}")

"""
Visualise alloy IE similarity directly in IE-space.

(A) Pairwise scatter grid (corner plot) for a selected set of alloys:
    each panel plots IE_alloyA vs IE_alloyB on the molecules measured on
    both alloys. Includes the y=x identity line and annotates Pearson r and n.
    Grid uses the 3 target alloys + MgCa to show 6 cluster-vs-cluster /
    cluster-vs-outlier comparisons.

(B) AZ31 vs all other alloys overlay: one scatter with AZ31 IE on x and
    every other alloy's IE on y, coloured by the y-axis alloy.

Outputs (figures/):
    fig_pairwise_scatter_grid.{png,pdf}
    fig_az31_vs_others.{png,pdf}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
})
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "ExCorrDatasetClean.csv"
FIG_OUT = ROOT / "tl_crossalloy" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)

ALLOYS_ALL = ['AM50','AZ31','AZ91','CPMg220ppmFe','CPMg342ppmFe','E21',
              'HPMg50ppmFe','HPMg51ppmFe','MgCa','WE43','ZE41']
SHORT = {"CPMg220ppmFe": "CPMg220", "CPMg342ppmFe": "CPMg342",
         "HPMg50ppmFe": "HPMg50", "HPMg51ppmFe": "HPMg51"}
def short(a): return SHORT.get(a, a)

df = pd.read_csv(DATA_CSV)
mg = df[(df['BaseMaterial'] == 'Mg') & (df['Alloy'].isin(ALLOYS_ALL))].copy()

# Per-alloy per-molecule mean IE on the *raw* IE scale for interpretable axes.
agg = mg.groupby(['Alloy', 'canonical_SMILES'])['IE'].mean().reset_index()
ie_by_alloy = {a: agg[agg['Alloy'] == a].set_index('canonical_SMILES')['IE']
               for a in ALLOYS_ALL}

def common_ie(a, b):
    common = ie_by_alloy[a].index.intersection(ie_by_alloy[b].index)
    if len(common) == 0:
        return np.array([]), np.array([])
    return ie_by_alloy[a].loc[common].values, ie_by_alloy[b].loc[common].values

# Visualisation range: most molecules are in the inhibition region (IE 0--100).
# A long tail of accelerator molecules extends to IE ~ -3000; we keep these in
# the correlation calculation but clip the scatter axes to keep plots legible.
VMIN, VMAX = -100, 100
def in_range(x, y):
    return (x >= VMIN) & (x <= VMAX) & (y >= VMIN) & (y <= VMAX)

# =============================================================================
# (A) Corner plot: 4 alloys (3 targets + MgCa) -> 6 unique pairs
# =============================================================================
GRID = ['AZ31', 'AZ91', 'WE43', 'MgCa']
n = len(GRID)

fig, axes = plt.subplots(n, n, figsize=(10.5, 10.5),
                         sharex=False, sharey=False)

cluster_color = "#2471A3"
mgca_color    = "#C0392B"

for i, a in enumerate(GRID):
    for j, b in enumerate(GRID):
        ax = axes[i, j]
        if i == j:
            ax.text(0.5, 0.5, short(a), ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    transform=ax.transAxes,
                    color=mgca_color if a == 'MgCa' else cluster_color)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            continue
        if i < j:
            # Upper triangle: leave empty (avoid duplicate info)
            ax.set_visible(False)
            continue
        # Lower triangle: scatter of IE(b on x) vs IE(a on y)
        x, y = common_ie(b, a)
        if len(x) < 3:
            ax.text(0.5, 0.5, "n<3", ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_xticks([]); ax.set_yticks([])
            continue
        col = mgca_color if (a == 'MgCa' or b == 'MgCa') else cluster_color
        # r computed on full data; only points in (VMIN, VMAX) shown
        r, _ = pearsonr(x, y)
        m = in_range(x, y)
        n_off = int((~m).sum())
        ax.scatter(x[m], y[m], s=22, alpha=0.55, color=col, edgecolor='none')
        ax.plot([VMIN, VMAX], [VMIN, VMAX], '--', color='gray', lw=1, alpha=0.6)
        annot = f"r = {r:+.2f}\nn = {len(x)}"
        if n_off:
            annot += f"\n({n_off} off-scale)"
        ax.text(0.04, 0.96, annot, ha='left', va='top',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))
        ax.set_xlim(VMIN - 5, VMAX + 5)
        ax.set_ylim(VMIN - 5, VMAX + 5)
        ax.tick_params(labelsize=8)

        # Axis labels only on outermost cells
        if j == 0:
            ax.set_ylabel(f"IE on {short(a)} (\\%)" if False else f"IE on {short(a)} (%)",
                          fontsize=10)
        else:
            ax.set_yticklabels([])
        if i == n - 1:
            ax.set_xlabel(f"IE on {short(b)} (%)", fontsize=10)
        else:
            ax.set_xticklabels([])

plt.tight_layout()
fig.savefig(FIG_OUT / "fig_pairwise_scatter_grid.png", dpi=300, bbox_inches='tight')
fig.savefig(FIG_OUT / "fig_pairwise_scatter_grid.pdf", bbox_inches='tight')
plt.close(fig)
print(f"Saved: {FIG_OUT/'fig_pairwise_scatter_grid.png'}")

# =============================================================================
# (B) AZ31 vs all other alloys overlay
# =============================================================================
X_ALLOY = 'AZ31'
others = [a for a in ALLOYS_ALL if a != X_ALLOY]

# Order the other alloys: commercial cluster first, then high-Fe, then MgCa
COMM = ['AM50','AZ91','E21','HPMg51ppmFe','WE43','ZE41']
HIFE = ['CPMg220ppmFe','CPMg342ppmFe','HPMg50ppmFe']
order = COMM + HIFE + ['MgCa']

# Distinct colors. Use tab10 for cluster, oranges for high-Fe, red for MgCa.
import matplotlib.cm as cm
cluster_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:len(COMM)]
hife_colors = plt.cm.Oranges(np.linspace(0.4, 0.85, len(HIFE)))
group_colors = {a: cluster_colors[i] for i, a in enumerate(COMM)}
group_colors.update({a: hife_colors[i] for i, a in enumerate(HIFE)})
group_colors['MgCa'] = "#C0392B"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0), sharey=True)

for ax, group_label, group_alloys in [
    (ax1, "Commercial cluster", COMM),
    (ax2, "High-Fe purity + MgCa contrast", HIFE + ['MgCa']),
]:
    for a in group_alloys:
        x, y = common_ie(X_ALLOY, a)
        if len(x) < 3:
            continue
        r, _ = pearsonr(x, y)  # r on full data
        m = in_range(x, y)
        ax.scatter(x[m], y[m], s=28, alpha=0.6, color=group_colors[a],
                   label=f"{short(a)} (n={len(x)}, r={r:+.2f})", edgecolor='none')
    ax.plot([VMIN, VMAX], [VMIN, VMAX], '--', color='gray', lw=1, alpha=0.5,
            label='y = x')
    ax.set_xlim(VMIN - 5, VMAX + 5); ax.set_ylim(VMIN - 5, VMAX + 5)
    ax.set_xlabel(f"IE on {X_ALLOY} (%) --- {group_label}", fontsize=11)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.85)
    ax.grid(alpha=0.25)

ax1.set_ylabel("IE on other alloy (%)", fontsize=11)
plt.tight_layout()
fig.savefig(FIG_OUT / "fig_az31_vs_others.png", dpi=300, bbox_inches='tight')
fig.savefig(FIG_OUT / "fig_az31_vs_others.pdf", bbox_inches='tight')
plt.close(fig)
print(f"Saved: {FIG_OUT/'fig_az31_vs_others.png'}")

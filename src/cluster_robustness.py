"""
Cluster-stability analysis for the SAR-similarity cluster claim.
For each pair of the 11 Mg alloys, compute:
  - n shared molecules (same canonical_SMILES)
  - Pearson r on per-molecule mean IE
  - 95% bootstrap CI on r (1000 resamples of molecule pairs)
  - Partial Pearson r' excluding top-decile-IE anchors (top 10% by mean IE)

Outputs:
  SI/table_cluster_robustness.csv  --- long format, all pairs
  SI/table_cluster_robustness_wide.csv --- r with CI, r_partial, n in wide form
  figures/fig_cluster_robustness.png/pdf --- r with CI error bars for the upper triangle
"""
import sys
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
from scipy.stats import pearsonr, spearmanr

# Correlation method used throughout this analysis (matches main-text Fig. 4).
CORR = spearmanr
CORR_NAME = "Spearman"

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "ExCorrDatasetClean.csv"
SI_OUT = ROOT / "tl_crossalloy" / "SI"
FIG_OUT = ROOT / "tl_crossalloy" / "figures"
SI_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

ALLOYS = ['AM50','AZ31','AZ91','CPMg220ppmFe','CPMg342ppmFe','E21',
          'HPMg50ppmFe','HPMg51ppmFe','MgCa','WE43','ZE41']

RNG = np.random.default_rng(42)
N_BOOT = 1000
TOP_FRAC = 0.10  # exclude top 10% IE molecules when testing anchor dependence

df = pd.read_csv(DATA_CSV)
mg = df[(df['BaseMaterial'] == 'Mg') & (df['Alloy'].isin(ALLOYS))].copy()

# Use transformed_IE to match the existing pearson_matrix.csv convention in analysis.py.
IE_COL = 'transformed_IE'
agg = (mg.groupby(['Alloy', 'canonical_SMILES'])[IE_COL]
         .mean()
         .reset_index())

ie_by_alloy = {a: agg[agg['Alloy'] == a].set_index('canonical_SMILES')[IE_COL]
               for a in ALLOYS}

rows = []
for i, a in enumerate(ALLOYS):
    for j, b in enumerate(ALLOYS):
        if i == j:
            rows.append(dict(alloy_a=a, alloy_b=b, n=len(ie_by_alloy[a]),
                             r=1.0, r_lo=1.0, r_hi=1.0,
                             r_partial=1.0, n_partial=len(ie_by_alloy[a])))
            continue
        common = ie_by_alloy[a].index.intersection(ie_by_alloy[b].index)
        n = len(common)
        if n < 3:
            rows.append(dict(alloy_a=a, alloy_b=b, n=n,
                             r=np.nan, r_lo=np.nan, r_hi=np.nan,
                             r_partial=np.nan, n_partial=0))
            continue
        x = ie_by_alloy[a].loc[common].values
        y = ie_by_alloy[b].loc[common].values
        r, _ = CORR(x, y)

        # Bootstrap over molecule pairs
        boots = np.empty(N_BOOT)
        idx = np.arange(n)
        for k in range(N_BOOT):
            pick = RNG.choice(idx, size=n, replace=True)
            xb = x[pick]; yb = y[pick]
            if xb.std() == 0 or yb.std() == 0:
                boots[k] = np.nan
            else:
                boots[k], _ = CORR(xb, yb)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])

        # Partial: drop top-TOP_FRAC IE molecules (by mean IE across the pair)
        mean_ie = (x + y) / 2.0
        cutoff = np.quantile(mean_ie, 1.0 - TOP_FRAC)
        keep = mean_ie < cutoff
        n_p = int(keep.sum())
        if n_p >= 3:
            xp = x[keep]; yp = y[keep]
            if xp.std() > 0 and yp.std() > 0:
                r_p, _ = CORR(xp, yp)
            else:
                r_p = np.nan
        else:
            r_p = np.nan

        rows.append(dict(alloy_a=a, alloy_b=b, n=n, r=r,
                         r_lo=lo, r_hi=hi,
                         r_partial=r_p, n_partial=n_p))

long = pd.DataFrame(rows)
long.to_csv(SI_OUT / "table_cluster_robustness.csv", index=False, float_format="%.4f")

# Compact long-format table for SI inclusion: one row per unique pair,
# columns: Pair | n | r | 95% CI | r' (anchor-excluded) | n'
# Upper triangle only (skip i==j and duplicates).
COMM_SET = set(['AM50','AZ31','AZ91','E21','HPMg51ppmFe','WE43','ZE41'])
def pair_kind(a, b):
    if a in COMM_SET and b in COMM_SET:
        return 'commercial'
    if 'MgCa' in (a, b):
        return 'MgCa'
    return 'other'

compact_rows = []
seen = set()
SHORT = {"CPMg220ppmFe": "CPMg220", "CPMg342ppmFe": "CPMg342",
         "HPMg50ppmFe": "HPMg50", "HPMg51ppmFe": "HPMg51"}
def short(a):
    return SHORT.get(a, a)

for a in ALLOYS:
    for b in ALLOYS:
        if a == b:
            continue
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        row = long[(long.alloy_a == a) & (long.alloy_b == b)].iloc[0]
        if np.isnan(row.r):
            continue
        compact_rows.append(dict(
            colpair=f"{short(a)}--{short(b)}",
            colkind=pair_kind(a, b),
            colnn=int(row.n),
            colrr=f"{row.r:.2f}",
            colci=f"[{row.r_lo:.2f}, {row.r_hi:.2f}]",
            colrp=f"{row.r_partial:.2f}" if not np.isnan(row.r_partial) else '—',
            colnp=int(row.n_partial),
        ))

compact = pd.DataFrame(compact_rows)
# Order: commercial first, then MgCa, then other
order_map = {'commercial': 0, 'MgCa': 1, 'other': 2}
compact['_sort'] = compact['colkind'].map(order_map)
compact = compact.sort_values(['_sort', 'colpair']).drop(columns=['_sort'])
# Keep the CSV around for inspection / data-availability.
compact[['colpair','colkind','colnn','colrr','colci','colrp','colnp']].to_csv(
    SI_OUT / "table_cluster_robustness_compact.csv", index=False)

# Also emit a ready-to-\input LaTeX tabular body --- avoids csvsimple
# parsing fragility on the supplement side.
tex_path = SI_OUT / "table_cluster_robustness_compact.tex"
with open(tex_path, "w", encoding="utf-8") as fh:
    for _, row in compact.iterrows():
        # Escape LaTeX-special chars conservatively (none present in our data).
        cells = [str(row['colpair']),
                 str(row['colkind']),
                 str(row['colnn']),
                 str(row['colrr']),
                 str(row['colci']),
                 str(row['colrp']),
                 str(row['colnp'])]
        fh.write(" & ".join(cells) + " \\\\\n")
print(f"Saved: {tex_path}")

# Commercial-cluster summary
COMM = ['AM50','AZ31','AZ91','E21','HPMg51ppmFe','WE43','ZE41']
mask = long.alloy_a.isin(COMM) & long.alloy_b.isin(COMM) & (long.alloy_a != long.alloy_b)
sub = long[mask]
print(f"Commercial cluster ({len(COMM)} alloys, {(len(COMM)*(len(COMM)-1))} ordered pairs):")
print(f"  r              : {sub.r.min():.3f} -- {sub.r.max():.3f}  (median {sub.r.median():.3f})")
print(f"  r 95% CI low   : {sub.r_lo.min():.3f} -- {sub.r_lo.max():.3f}")
print(f"  r 95% CI high  : {sub.r_hi.min():.3f} -- {sub.r_hi.max():.3f}")
print(f"  r_partial      : {sub.r_partial.min():.3f} -- {sub.r_partial.max():.3f}  (median {sub.r_partial.median():.3f})")
print(f"  n shared mols  : {sub.n.min()} -- {sub.n.max()}  (median {int(sub.n.median())})")

# MgCa summary
mask_mc = (long.alloy_a == 'MgCa') & (long.alloy_b != 'MgCa')
submc = long[mask_mc]
print(f"\nMgCa vs all others:")
print(f"  r              : {submc.r.min():.3f} -- {submc.r.max():.3f}  (median {submc.r.median():.3f})")
print(f"  r 95% CI       : low in [{submc.r_lo.min():.3f}, {submc.r_lo.max():.3f}], high in [{submc.r_hi.min():.3f}, {submc.r_hi.max():.3f}]")
print(f"  r_partial      : {submc.r_partial.min():.3f} -- {submc.r_partial.max():.3f}")
print(f"  n shared mols  : {submc.n.min()} -- {submc.n.max()}")

# ---- Figure: horizontal forest plot of pairwise Spearman r with CIs ----
def _rows(pairs):
    out = []
    for a, b in pairs:
        row = long[(long.alloy_a == a) & (long.alloy_b == b)]
        if not len(row):
            row = long[(long.alloy_a == b) & (long.alloy_b == a)]
        rr = row.iloc[0]
        out.append((f"{short(a)}–{short(b)}", rr.r, rr.r_lo, rr.r_hi, rr.r_partial))
    return out

comm_pairs = [(a, b) for i, a in enumerate(COMM) for b in COMM[i+1:]]
cr = sorted(_rows(comm_pairs), key=lambda t: t[1], reverse=True)
mc_pairs = [(a, 'MgCa') for a in ALLOYS if a != 'MgCa']
mcr = sorted(_rows(mc_pairs), key=lambda t: t[1], reverse=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for ax, rows, ttl in [(ax1, cr, 'Commercial-cluster pairs'),
                      (ax2, mcr, 'MgCa vs. every other alloy')]:
    lbl = [t[0] for t in rows]
    rr = np.array([t[1] for t in rows]); lo = np.array([t[2] for t in rows])
    hi = np.array([t[3] for t in rows]); rp = np.array([t[4] for t in rows])
    yy = np.arange(len(rows))[::-1]
    ax.axvspan(0.67, 1.0, color='#2ecc71', alpha=0.10, lw=0)
    ax.hlines(yy, lo, hi, color='#9bbcd6', lw=2.5, zorder=2)
    ax.scatter(rr, yy, color='#2471A3', s=42, zorder=3, label=f'{CORR_NAME} r (95% CI)')
    ax.scatter(rp, yy, facecolor='none', edgecolor='#C0392B', s=42, marker='D',
               zorder=4, label=f"r' (top-{int(TOP_FRAC*100)}% IE excl.)")
    ax.axvline(0.67, color='#27ae60', ls='--', lw=1)
    ax.axvline(0, color='gray', lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(lbl, fontsize=8)
    ax.set_xlabel(f"{CORR_NAME} r"); ax.set_title(ttl, fontsize=11)
    ax.grid(True, axis='x', alpha=0.25, ls='--')
    ax.spines[['top', 'right']].set_visible(False)
ax1.set_xlim(-0.2, 1.05); ax2.set_xlim(-0.45, 1.05)
# Commercial pairs all have r >= 0.55, so the lower-left corner is empty.
ax1.legend(loc='lower left', fontsize=9, framealpha=0.95, bbox_to_anchor=(0.01, 0.01))
plt.tight_layout()
fig.savefig(FIG_OUT / "fig_cluster_robustness.png", dpi=300, bbox_inches='tight')
fig.savefig(FIG_OUT / "fig_cluster_robustness.pdf", bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {SI_OUT/'table_cluster_robustness.csv'}")
print(f"Saved: {FIG_OUT/'fig_cluster_robustness.png'}")

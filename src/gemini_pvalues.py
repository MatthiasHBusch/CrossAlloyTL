"""
Verify the < 10^-6 Fisher-combined p-values for Gemini and add p-values for
the zero-shot probe.

Approach: load molecule-level predictions from results/gemini/ and
results/gemini_zeroshot/, average over iterations per molecule, pair with
ground-truth IE, and compute Pearson r and its two-sided p-value on the
POOLED prediction-vs-truth set (n = 75 per alloy, n = 225 pooled across
the three target alloys).

Why pool: the per-fold Pearson test on n_test = 15 has only 13 degrees of
freedom and limited power; pooling per-molecule predictions across folds
gives one test on n = 75 (per alloy) or n = 225 (pooled), which is the
strongest single statement we can make about whether Gemini's apparent
correlation is significant.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
GEMINI_DIR = ROOT / "results" / "gemini"
ZS_DIR = ROOT / "results" / "gemini_zeroshot"
DATA_CSV = ROOT / "data" / "ExCorrDatasetClean.csv"
SI = ROOT / "SI"

ALLOYS = ["AZ31", "AZ91", "WE43"]
SETTINGS = ["exact", "close_filt", "close_unfilt", "far_filt", "far_unfilt"]


def canonical(smi):
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else smi
    except Exception:
        return smi


# Build per-alloy IUPAC -> mean(IE) lookup (matches what parse_gemini_results does).
data = pd.read_csv(DATA_CSV)
data_mg = data[data["BaseMaterial"] == "Mg"]
ie_lookup = {
    a: data_mg[data_mg["Alloy"] == a].groupby("IUPAC")["IE"].mean().to_dict()
    for a in ALLOYS
}


def load_incontext(alloy, setting):
    """Return list of (mol, pred, truth) tuples for this (alloy, setting)."""
    fp = GEMINI_DIR / f"Gemini3_1Pro_{alloy}_{setting}.json"
    if not fp.exists():
        return []
    with open(fp) as f:
        d = json.load(f)
    pairs = []
    # Walk: input_type -> approach -> model -> num_exact -> num_close -> num_far -> mol -> [preds]
    for _, it in d.items():
        for _, app in it.items():
            for _, mo in app.items():
                for _, ne in mo.items():
                    for _, nc in ne.items():
                        for _, mols in nc.items():
                            if not mols:
                                continue
                            for mol, preds in mols.items():
                                cleaned = [float(p) for p in preds
                                           if p is not None and isinstance(p, (int, float, str))
                                           and str(p).replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit() or
                                           (isinstance(p, (int, float)))]
                                cleaned = [float(p) for p in preds
                                           if isinstance(p, (int, float)) and np.isfinite(p)]
                                if not cleaned or mol not in ie_lookup[alloy]:
                                    continue
                                pairs.append((mol, float(np.mean(cleaned)),
                                              float(ie_lookup[alloy][mol])))
    return pairs


def load_zeroshot(alloy):
    fp = ZS_DIR / f"Gemini3_1Pro_{alloy}_zeroshot.json"
    if not fp.exists():
        return []
    with open(fp) as f:
        d = json.load(f)
    pairs = []
    for mol, preds in d["predictions"].items():
        cleaned = [float(p) for p in preds
                   if isinstance(p, (int, float)) and np.isfinite(p)]
        if not cleaned:
            continue
        truth = d["ground_truth"].get(mol)
        if truth is None or not np.isfinite(truth):
            continue
        pairs.append((mol, float(np.mean(cleaned)), float(truth)))
    return pairs


def pearson_with_p(pairs):
    if len(pairs) < 3:
        return float("nan"), float("nan"), len(pairs)
    y = np.array([t[2] for t in pairs])
    p = np.array([t[1] for t in pairs])
    if y.std() == 0 or p.std() == 0:
        return float("nan"), float("nan"), len(pairs)
    r, pv = pearsonr(p, y)
    return float(r), float(pv), len(pairs)


def fmt_p(p):
    if not np.isfinite(p):
        return "---"
    if p < 1e-12:
        return r"$<\!10^{-12}$"
    if p < 1e-6:
        exp = int(np.floor(np.log10(p)))
        return rf"$\sim\!10^{{{exp}}}$"
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        return rf"$\sim\!10^{{{exp}}}$"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


# ----------------------------------------------------------------------------
# In-context Gemini: per-setting pooled across 3 alloys
# ----------------------------------------------------------------------------
print(f"{'Setting':<14} {'per-alloy r/p':<45} {'pooled (n=225) r/p'}")
print("-" * 100)
ic_results = {}
for setting in SETTINGS:
    perAlloy = []
    pooled = []
    for alloy in ALLOYS:
        pairs = load_incontext(alloy, setting)
        if not pairs:
            perAlloy.append((alloy, "-"))
            continue
        r, p, n = pearson_with_p(pairs)
        perAlloy.append((alloy, f"r={r:+.2f} p={p:.2g} n={n}"))
        pooled.extend(pairs)
    if pooled:
        r_p, p_p, n_p = pearson_with_p(pooled)
        ic_results[setting] = (r_p, p_p, n_p)
        ic_str = f"r={r_p:+.3f} p={p_p:.2e} n={n_p}"
    else:
        ic_str = "no data"
    print(f"{setting:<14} " +
          " | ".join(f"{a}: {s}" for a, s in perAlloy) +
          f"   --> {ic_str}")

# ----------------------------------------------------------------------------
# Zero-shot: per-alloy and pooled
# ----------------------------------------------------------------------------
print()
print(f"{'Zero-shot':<14} {'per-alloy r/p':<45} {'pooled (n=225) r/p'}")
print("-" * 100)
zs_per_alloy = []
pooled = []
for alloy in ALLOYS:
    pairs = load_zeroshot(alloy)
    if not pairs:
        zs_per_alloy.append((alloy, None, None, 0))
        continue
    r, p, n = pearson_with_p(pairs)
    zs_per_alloy.append((alloy, r, p, n))
    pooled.extend(pairs)
zs_pooled = pearson_with_p(pooled) if pooled else (np.nan, np.nan, 0)

for a, r, p, n in zs_per_alloy:
    if r is None:
        print(f"  {a}: no data")
    else:
        print(f"  {a}: r={r:+.3f} p={p:.3e} n={n}")
print(f"  pooled: r={zs_pooled[0]:+.3f} p={zs_pooled[1]:.3e} n={zs_pooled[2]}")

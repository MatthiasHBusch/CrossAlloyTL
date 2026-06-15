"""
Compute every aggregate number quoted in the manuscript and SI from the
authoritative result files (results/experiment_results.csv etc.), so that
text, figures and SI tables all come from one source.

Outputs a readable report to stdout.
"""
import sys, os, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "experiment_results.csv").dropna(subset=["pearson_r"])

FITTED = ["RF", "GBR", "MLP", "kNN_Tan", "ChemProp"]
CALIBRATED = ["RF", "GBR", "MLP", "ChemProp"]
SETTINGS = ["exact", "close_filt", "close_unfilt", "far_filt", "far_unfilt"]
MODELS = FITTED + ["Gemini3.1Pro"]

print("=" * 80)
print("Row coverage (model x setting):")
print(df.groupby(["model", "setting"]).size().unstack(fill_value=0))

def cell(model, setting, metric="r2"):
    sub = df[(df["model"] == model) & (df["setting"] == setting)][metric]
    return sub.mean(), sub.std() / np.sqrt(len(sub)), len(sub)

print("\n" + "=" * 80)
for metric, fmt in [("r2", "+.2f"), ("pearson_r", "+.2f"), ("mae", ".0f")]:
    print(f"\n--- {metric}: mean +/- SE per (model, setting) ---")
    header = f"{'model':<14}" + "".join(f"{s:>22}" for s in SETTINGS)
    print(header)
    for m in MODELS:
        row = f"{m:<14}"
        for s in SETTINGS:
            mu, se, n = cell(m, s, metric)
            row += f"{format(mu, fmt):>12} ± {se:<7.2f}"
        print(row)
    # means
    for label, group in [("mean(5 fitted)", FITTED), ("mean(4 calib)", CALIBRATED)]:
        row = f"{label:<14}"
        for s in SETTINGS:
            mus = [cell(m, s, metric)[0] for m in group]
            row += f"{format(np.mean(mus), fmt):>12}          "
        print(row)

print("\n" + "=" * 80)
print("Leakage deltas (unfilt - filt) on R2, per model:")
for scope in ["close", "far"]:
    print(f"  {scope}:")
    deltas = []
    for m in FITTED:
        d = cell(m, f"{scope}_unfilt")[0] - cell(m, f"{scope}_filt")[0]
        deltas.append(d)
        print(f"    {m:<10} {d:+.2f}")
    print(f"    mean(5 fitted): {np.mean(deltas):+.2f}")
    g = cell("Gemini3.1Pro", f"{scope}_unfilt")[0] - cell("Gemini3.1Pro", f"{scope}_filt")[0]
    print(f"    Gemini:     {g:+.2f}")

print("\n" + "=" * 80)
print("Exact -> close_unfilt shift on R2:")
for label, group in [("5 fitted", FITTED), ("4 calibrated", CALIBRATED)]:
    e = np.mean([cell(m, "exact")[0] for m in group])
    cu = np.mean([cell(m, "close_unfilt")[0] for m in group])
    fu = np.mean([cell(m, "far_unfilt")[0] for m in group])
    cf = np.mean([cell(m, "close_filt")[0] for m in group])
    ff = np.mean([cell(m, "far_filt")[0] for m in group])
    print(f"  {label}: exact={e:+.2f}  close_filt={cf:+.2f}  close_unfilt={cu:+.2f}"
          f"  far_filt={ff:+.2f}  far_unfilt={fu:+.2f}  shift={cu-e:+.2f}")

print("\nBest model per setting (R2):")
for s in SETTINGS:
    vals = {m: cell(m, s)[0] for m in MODELS}
    best = max(vals, key=vals.get)
    print(f"  {s:<14} best={best} ({vals[best]:+.2f});  " +
          "  ".join(f"{m}={v:+.2f}" for m, v in vals.items()))

print("\n" + "=" * 80)
print("Per-alloy close_unfilt R2 (Table S6 check):")
for alloy in ["AZ31", "AZ91", "WE43"]:
    row = f"  {alloy}: "
    for m in FITTED:
        sub = df[(df.model == m) & (df.setting == "close_unfilt") & (df.alloy == alloy)]["r2"]
        row += f"{m}={sub.mean():+.2f}±{sub.std()/np.sqrt(len(sub)):.2f}  "
    print(row)
for alloy in ["AZ31", "AZ91", "WE43"]:
    cal = np.mean([df[(df.model == m) & (df.setting == "close_unfilt") & (df.alloy == alloy)]["r2"].mean()
                   for m in CALIBRATED])
    print(f"  {alloy} mean(4 calibrated) = {cal:+.2f}")

print("\nGemini per-alloy per-setting R2:")
for alloy in ["AZ31", "AZ91", "WE43"]:
    row = f"  {alloy}: "
    for s in SETTINGS:
        sub = df[(df.model == "Gemini3.1Pro") & (df.setting == s) & (df.alloy == alloy)]["r2"]
        row += f"{s}={sub.mean():+.2f}  "
    print(row)

print("\n" + "=" * 80)
print("Wilcoxon paired tests over 15 (alloy,fold): exact vs close_filt")
for metric in ["pearson_r", "r2"]:
    print(f"  metric={metric}")
    pooled_e, pooled_c = [], []
    for m in FITTED:
        e = df[(df.model == m) & (df.setting == "exact")].sort_values(["alloy", "fold"])[metric].values
        c = df[(df.model == m) & (df.setting == "close_filt")].sort_values(["alloy", "fold"])[metric].values
        stat, p = wilcoxon(e, c)
        pooled_e.extend(e.tolist()); pooled_c.extend(c.tolist())
        print(f"    {m:<10} p={p:.2f}")
    stat, p = wilcoxon(pooled_e, pooled_c)
    print(f"    pooled     p={p:.2f}")

# ---- MgCa contrast --------------------------------------------------------
mg = pd.read_csv(ROOT / "results" / "mgca_contrast.csv").dropna(subset=["pearson_r"])
print("\n" + "=" * 80)
print("MgCa contrast (4 folds):")
print("Coverage:", mg.groupby(["model", "setting"]).size().to_dict())
for metric, fmt in [("r2", "+.2f"), ("pearson_r", "+.2f"), ("mae", ".1f")]:
    print(f"\n--- MgCa {metric} mean ± SE ---")
    for m in sorted(mg.model.unique()):
        row = f"  {m:<14}"
        for s in SETTINGS:
            sub = mg[(mg.model == m) & (mg.setting == s)][metric]
            if len(sub) == 0:
                row += f"{'--':>10}        "
            else:
                row += f"{format(sub.mean(), fmt):>10} ± {sub.std()/np.sqrt(len(sub)):<6.2f}"
        print(row)
print("\nMgCa: mean over 4 calibrated, exact:",
      np.mean([mg[(mg.model == m) & (mg.setting == 'exact')]["r2"].mean() for m in CALIBRATED]))
print("MgCa: mean over 4 calibrated, close_unfilt:",
      np.mean([mg[(mg.model == m) & (mg.setting == 'close_unfilt')]["r2"].mean() for m in CALIBRATED]))
print("MgCa exact R2 range over fitted:",
      [f"{m}={mg[(mg.model == m) & (mg.setting == 'exact')]['r2'].mean():+.2f}" for m in FITTED])
print("MgCa close_unfilt R2 over fitted:",
      [f"{m}={mg[(mg.model == m) & (mg.setting == 'close_unfilt')]['r2'].mean():+.2f}" for m in FITTED])
print("MgCa far_unfilt/far_filt R2 over fitted:",
      [f"{m}: fu={mg[(mg.model == m) & (mg.setting == 'far_unfilt')]['r2'].mean():+.2f}"
       f" ff={mg[(mg.model == m) & (mg.setting == 'far_filt')]['r2'].mean():+.2f}" for m in FITTED])

# ---- Zero-shot ------------------------------------------------------------
print("\n" + "=" * 80)
print("Gemini zero-shot:")
from sklearn.metrics import r2_score, mean_absolute_error
all_y, all_p = [], []
for f in sorted(glob.glob(str(ROOT / "results" / "gemini_zeroshot" / "*.json"))):
    d = json.load(open(f))
    ys, ps = [], []
    for mol, plist in d["predictions"].items():
        cleaned = [p for p in plist if p is not None and isinstance(p, (int, float)) and np.isfinite(p)]
        if not cleaned:
            continue
        ys.append(d["ground_truth"][mol]); ps.append(np.mean(cleaned))
    ys, ps = np.array(ys), np.array(ps)
    r, p = pearsonr(ys, ps)
    print(f"  {d['alloy']}: n={len(ys)} r={r:+.2f} p={p:.3f} R2={r2_score(ys, ps):+.2f} "
          f"MAE={mean_absolute_error(ys, ps):.0f}")
    all_y.extend(ys.tolist()); all_p.extend(ps.tolist())
ay, ap = np.array(all_y), np.array(all_p)
r, p = pearsonr(ay, ap)
print(f"  pooled: n={len(ay)} r={r:+.2f} p={p:.4f} R2={r2_score(ay, ap):+.2f} "
      f"MAE={mean_absolute_error(ay, ap):.0f}")

# ---- Sample counts --------------------------------------------------------
print("\n" + "=" * 80)
print("Sample counts (results/sample_counts.csv):")
sc = pd.read_csv(ROOT / "results" / "sample_counts.csv")
print(sc.to_string())
print("\nTest/pool overlap (results/test_pool_overlap.csv):")
ov = pd.read_csv(ROOT / "results" / "test_pool_overlap.csv")
print(ov.to_string())
print("\nOverlap means:", ov.mean(numeric_only=True).to_dict())

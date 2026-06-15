"""
Aggregate per-fold molecule-level predictions in results/predictions/*.json
into pooled Pearson r + p per (model, setting), per-alloy (n=75) and
across-all-alloys (n=225).

Also merges in Gemini 3.1 Pro's pooled stats (computed from the molecule-level
JSONs in results/gemini/ via gemini_pvalues.py).

Outputs:
  SI/table_pearson_r_pvalues_pooled.csv   (full per-(model, setting) table)
  SI/table_pearson_r_pvalues_pooled.tex   (ready-to-paste LaTeX body)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results" / "predictions"
GEMINI_DIR = ROOT / "results" / "gemini"
DATA_CSV = ROOT / "data" / "ExCorrDatasetClean.csv"
SI = ROOT / "SI"

ALLOYS = ["AZ31", "AZ91", "WE43"]
SETTINGS = ["exact", "close_filt", "close_unfilt", "far_filt", "far_unfilt"]
MODELS = ["RF", "GBR", "MLP", "kNN_Tan", "ChemProp"]

SETTING_LABEL = {"exact": "Exact", "close_filt": "Close filt.",
                 "close_unfilt": "Close unfilt.",
                 "far_filt": "Far filt.", "far_unfilt": "Far unfilt."}
MODEL_LABEL = {"RF": "RF", "GBR": "GBR", "MLP": "MLP",
               "kNN_Tan": "kNN-Tanimoto", "ChemProp": "ChemProp",
               "Gemini3.1Pro": "Gemini 3.1 Pro"}


def fmt_p(p):
    if not np.isfinite(p):
        return "---"
    if p < 1e-50:
        return r"$<\!10^{-50}$"
    if p < 1e-12:
        exp = int(np.floor(np.log10(p)))
        return rf"$\sim\!10^{{{exp}}}$"
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        return rf"$\sim\!10^{{{exp}}}$"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def pooled_for_fitted(model, setting):
    """Combine all (alloy, fold) predictions in PRED_DIR for this (model, setting)."""
    y_all, p_all = [], []
    for alloy in ALLOYS:
        for fold in range(5):
            fp = PRED_DIR / f"{alloy}_f{fold}_{setting}_{model}.json"
            if not fp.exists():
                continue
            with open(fp) as fh:
                d = json.load(fh)
            y_all.extend(d["y_test"])
            p_all.extend(d["y_pred"])
    if len(y_all) < 3:
        return float("nan"), float("nan"), 0
    y = np.array(y_all); p = np.array(p_all)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if y.std() == 0 or p.std() == 0:
        return float("nan"), float("nan"), len(y)
    r, pv = pearsonr(p, y)
    return float(r), float(pv), len(y)


def pooled_for_gemini(setting):
    """Pool molecule-level Gemini predictions from results/gemini/."""
    df_data = pd.read_csv(DATA_CSV)
    df_mg = df_data[df_data["BaseMaterial"] == "Mg"]
    ie_lookup = {
        a: df_mg[df_mg["Alloy"] == a].groupby("IUPAC")["IE"].mean().to_dict()
        for a in ALLOYS
    }
    y_all, p_all = [], []
    for alloy in ALLOYS:
        fp = GEMINI_DIR / f"Gemini3_1Pro_{alloy}_{setting}.json"
        if not fp.exists():
            continue
        with open(fp) as fh:
            d = json.load(fh)
        for _, it in d.items():
            for _, app in it.items():
                for _, mo in app.items():
                    for _, ne in mo.items():
                        for _, nc in ne.items():
                            for _, mols in nc.items():
                                if not mols:
                                    continue
                                for mol, preds in mols.items():
                                    cleaned = [float(v) for v in preds
                                               if isinstance(v, (int, float))
                                               and np.isfinite(v)]
                                    if not cleaned or mol not in ie_lookup[alloy]:
                                        continue
                                    y_all.append(ie_lookup[alloy][mol])
                                    p_all.append(float(np.mean(cleaned)))
    if len(y_all) < 3:
        return float("nan"), float("nan"), 0
    y = np.array(y_all); p = np.array(p_all)
    r, pv = pearsonr(p, y)
    return float(r), float(pv), len(y)


rows = []
for model in MODELS + ["Gemini3.1Pro"]:
    for setting in SETTINGS:
        if model == "Gemini3.1Pro":
            r, p, n = pooled_for_gemini(setting)
        else:
            r, p, n = pooled_for_fitted(model, setting)
        rows.append({
            "model": MODEL_LABEL[model],
            "setting": SETTING_LABEL[setting],
            "r": r, "p": p, "n": n,
        })

df = pd.DataFrame(rows)
df.to_csv(SI / "table_pearson_r_pvalues_pooled.csv", index=False, float_format="%.4g")

# Wide LaTeX rows: per model, p across 5 settings.
SI_TEX = SI / "table_pearson_r_pvalues_pooled.tex"
lines = []
for model in MODELS + ["Gemini3.1Pro"]:
    label = MODEL_LABEL[model]
    cells = []
    for setting in SETTINGS:
        sub = df[(df.model == label) & (df.setting == SETTING_LABEL[setting])]
        if sub.empty:
            cells.append("---")
        else:
            row = sub.iloc[0]
            if not np.isfinite(row.p):
                cells.append("---")
            else:
                cells.append(fmt_p(row.p))
    # Drop trailing \\ on the very last row so \bottomrule lands cleanly.
    suffix = "" if model == MODELS[-1] + "Gemini3.1Pro" else " \\\\"
    lines.append(label.replace("kNN-Tanimoto", "kNN-Tan")
                 + " & " + " & ".join(cells))
body = " \\\\\n".join(lines) + "\n"
SI_TEX.write_text(body, encoding="utf-8")

print(df.pivot(index="model", columns="setting", values="p").to_string())
print()
print("n per cell:")
print(df.pivot(index="model", columns="setting", values="n").to_string())
print()
print(f"Saved: {SI / 'table_pearson_r_pvalues_pooled.csv'}")
print(f"Saved: {SI_TEX}")

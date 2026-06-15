"""
Fill numerical placeholders in manuscript.tex with computed values from
results/experiment_results.csv and results/mgca_contrast.csv.

Reads:  results/experiment_results.csv, results/mgca_contrast.csv
Writes: manuscript/manuscript_filled.tex (copy of manuscript.tex with
        \texttt{PLACEHOLDER} tokens replaced).

Usage:
    C:/Users/mbusc/miniconda3/envs/xtb_env/python.exe fill_placeholders.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
MAIN_CSV = ROOT / "results" / "experiment_results.csv"
MGCA_CSV = ROOT / "results" / "mgca_contrast.csv"
TEX_IN = ROOT / "manuscript" / "manuscript.tex"
TEX_OUT = ROOT / "manuscript" / "manuscript_filled.tex"


def fmt_r(v):
    """Format Pearson r with sign, 2 decimals."""
    return f"{v:+.2f}"


def fmt_delta(v):
    """Format Δ with sign, 2 decimals."""
    return f"{v:+.2f}"


def compute_placeholders():
    d = pd.read_csv(MAIN_CSV).dropna(subset=["pearson_r"])
    p = {}

    # Per-model means at the key settings
    for (model, setting), g in d.groupby(["model", "setting"]):
        r_mean = g["pearson_r"].mean()
        p[f"{model}_{setting}_r"] = fmt_r(r_mean)

    # Setting-level aggregates (averaged across all models)
    for setting, g in d.groupby("setting"):
        p[f"{setting.upper()}_R"] = fmt_r(g["pearson_r"].mean())

    # Best model at each key setting
    for setting in ["exact", "close_unfilt", "close_filt", "far_unfilt", "far_filt"]:
        sub = d[d["setting"] == setting]
        if len(sub) == 0: continue
        model_means = sub.groupby("model")["pearson_r"].mean().sort_values(ascending=False)
        best_model = model_means.index[0]
        p[f"BEST_{setting.upper()}_R"] = fmt_r(model_means.iloc[0])
        p[f"BEST_{setting.upper()}_MODEL"] = best_model

    # ChemProp and RF / kNN specifics
    for model in ["ChemProp", "RF", "kNN_Tan"]:
        for setting in ["exact", "close_unfilt", "close_filt"]:
            sub = d[(d["model"] == model) & (d["setting"] == setting)]
            if len(sub) > 0:
                p[f"{model.upper()}_{setting.upper()}_R"] = fmt_r(
                    sub["pearson_r"].mean())
    # Short aliases for abstract/body
    for short, long in [("CP_CLOSE_R", ("ChemProp", "close_unfilt")),
                        ("RF_CLOSE_R", ("RF", "close_unfilt")),
                        ("KNN_CLOSE_R", ("kNN_Tan", "close_unfilt"))]:
        sub = d[(d["model"] == long[0]) & (d["setting"] == long[1])]
        if len(sub) > 0:
            p[short] = fmt_r(sub["pearson_r"].mean())

    # Leakage delta per model (close)
    deltas = {}
    for model in d["model"].unique():
        unf = d[(d["model"] == model) & (d["setting"] == "close_unfilt")]["pearson_r"].mean()
        filt = d[(d["model"] == model) & (d["setting"] == "close_filt")]["pearson_r"].mean()
        if np.isnan(unf) or np.isnan(filt):
            continue
        deltas[model] = unf - filt
    if deltas:
        p["DELTA_CLOSE_AVG"] = fmt_delta(np.mean(list(deltas.values())))
        dmax = max(deltas, key=deltas.get)
        dmin = min(deltas, key=deltas.get)
        p["DELTA_MAX"] = fmt_delta(deltas[dmax])
        p["MODEL_MAX"] = dmax
        p["DELTA_MIN"] = fmt_delta(deltas[dmin])
        p["MODEL_MIN"] = dmin

    # Model range at close_unfilt
    sub_cu = d[d["setting"] == "close_unfilt"].groupby("model")["pearson_r"].mean()
    if len(sub_cu) > 0:
        p["MODEL_RANGE"] = f"{fmt_r(sub_cu.min())}\\text{{--}}{fmt_r(sub_cu.max())}"

    # Wilcoxon exact vs close_filt (paired by (alloy, fold))
    d_ex = d[d["setting"] == "exact"].set_index(["alloy", "fold", "model"])["pearson_r"]
    d_cf = d[d["setting"] == "close_filt"].set_index(["alloy", "fold", "model"])["pearson_r"]
    common = d_ex.index.intersection(d_cf.index)
    if len(common) > 5:
        diffs = (d_cf.loc[common] - d_ex.loc[common]).values
        if np.any(diffs != 0):
            try:
                _, pv = wilcoxon(diffs)
                p["WILC_PVAL"] = f"{pv:.3f}"
            except Exception:
                p["WILC_PVAL"] = "n/a"
        else:
            p["WILC_PVAL"] = "1.000"

    # MgCa contrast
    if MGCA_CSV.exists():
        m = pd.read_csv(MGCA_CSV).dropna(subset=["pearson_r"])
        if len(m) > 0:
            best_cu = m[m["setting"] == "close_unfilt"]["pearson_r"].mean()
            p["MGCA_BEST"] = fmt_r(best_cu)
            unf = m[m["setting"] == "close_unfilt"]["pearson_r"].mean()
            filt = m[m["setting"] == "close_filt"]["pearson_r"].mean()
            if not np.isnan(unf) and not np.isnan(filt):
                p["MGCA_DELTA"] = fmt_delta(unf - filt)

    # Target best
    best_target = d[d["setting"] == "close_unfilt"]["pearson_r"].max()
    p["AZ_BEST"] = fmt_r(best_target) if not np.isnan(best_target) else "+0.00"

    # Overall TL/FILT/NT for abstract
    p["TL"] = p.get("CLOSE_UNFILT_R", "+0.00")
    p["FILT"] = p.get("CLOSE_FILT_R", "+0.00")
    p["NT"] = p.get("EXACT_R", "+0.00")

    return p


def fill(placeholders):
    tex = TEX_IN.read_text(encoding="utf-8")
    # Also add pre-escaped variants like CLOSE\_UNFILT\_R (LaTeX escape)
    escaped = {}
    for k, v in placeholders.items():
        if "_" in k:
            escaped[k.replace("_", "\\_")] = v
    all_replacements = {**placeholders, **escaped}
    for key, value in sorted(all_replacements.items(),
                             key=lambda kv: -len(kv[0])):
        tex = tex.replace(f"\\texttt{{{key}}}", value)
    # Also replace legacy placeholder names
    tex = tex.replace(r"\mathrm{TL}", placeholders.get("TL", "TL"))
    tex = tex.replace(r"\mathrm{FILT}", placeholders.get("FILT", "FILT"))
    tex = tex.replace(r"\mathrm{NT}", placeholders.get("NT", "NT"))
    TEX_OUT.write_text(tex, encoding="utf-8")


def main():
    if not MAIN_CSV.exists():
        print(f"Main CSV not found: {MAIN_CSV}")
        sys.exit(1)
    p = compute_placeholders()
    print(f"Computed {len(p)} placeholders:")
    for k in sorted(p.keys()):
        print(f"  {k:30s} = {p[k]}")
    fill(p)
    print(f"\nFilled manuscript written to {TEX_OUT}")


if __name__ == "__main__":
    main()

"""
Per-(model, setting) Pearson r summary with combined p-values.

For each cell, aggregate the 15 per-fold Pearson r values (3 target alloys x
5 folds) as:
  - mean r +/- SE
  - combined p-value via Fisher's method (chi^2 with 2k df)
  - fraction of folds with p < 0.05

Outputs:
  SI/table_pearson_r_with_p.tex   --- LaTeX-ready tabular body
  SI/table_pearson_r_with_p.csv   --- machine-readable copy
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, t as student_t


def pearson_p_from_r_n(r, n):
    """Two-sided p-value of the Pearson test given r and sample size n."""
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    mask = np.isfinite(r) & (n > 2) & (np.abs(r) < 1.0)
    p = np.full_like(r, np.nan)
    t_stat = r[mask] * np.sqrt(n[mask] - 2) / np.sqrt(1 - r[mask] ** 2)
    p[mask] = 2.0 * (1.0 - student_t.cdf(np.abs(t_stat), df=n[mask] - 2))
    return p

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "experiment_results.csv"
SI = ROOT / "SI"

SETTINGS_ORDER = ["exact", "close_filt", "close_unfilt", "far_filt", "far_unfilt"]
SETTING_LABEL = {
    "exact": "Exact", "close_filt": "Close filt.", "close_unfilt": "Close unfilt.",
    "far_filt": "Far filt.", "far_unfilt": "Far unfilt.",
}
MODEL_ORDER = ["RF", "GBR", "MLP", "kNN_Tan", "ChemProp", "Gemini3.1Pro"]
MODEL_LABEL = {"RF": "RF", "GBR": "GBR", "MLP": "MLP",
               "kNN_Tan": "kNN-Tanimoto", "ChemProp": "ChemProp",
               "Gemini3.1Pro": "Gemini 3.1 Pro"}


def fisher_combined_p(pvals):
    """Fisher's method: -2 sum(log p) ~ chi^2(2k); returns combined p."""
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[np.isfinite(pvals)]
    if len(pvals) == 0:
        return np.nan
    # Avoid log(0).
    pvals = np.clip(pvals, 1e-300, 1.0)
    chi = -2.0 * np.sum(np.log(pvals))
    df = 2 * len(pvals)
    return float(1.0 - chi2.cdf(chi, df))


def fmt_p(p):
    if not np.isfinite(p):
        return "---"
    if p < 1e-6:
        return r"$<\!10^{-6}$"
    if p < 1e-3:
        # Round to nearest decade.
        exp = int(np.floor(np.log10(p)))
        return rf"$\sim\!10^{{{exp}}}$"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


df = pd.read_csv(RESULTS).dropna(subset=["pearson_r"])

# Some models (e.g. Gemini) lack a stored pearson_p column; compute it from
# the per-fold r and n_test.
missing = df["pearson_p"].isna()
if missing.any():
    df.loc[missing, "pearson_p"] = pearson_p_from_r_n(
        df.loc[missing, "pearson_r"].values,
        df.loc[missing, "n_test"].values,
    )

# Wide format: rows = models, columns = settings, cells = Fisher-combined p.
wide = pd.DataFrame(index=[MODEL_LABEL[m] for m in MODEL_ORDER],
                    columns=[SETTING_LABEL[s] for s in SETTINGS_ORDER])
for model in MODEL_ORDER:
    for setting in SETTINGS_ORDER:
        sub = df[(df.model == model) & (df.setting == setting)]
        if len(sub) == 0:
            continue
        wide.loc[MODEL_LABEL[model], SETTING_LABEL[setting]] = fisher_combined_p(
            sub["pearson_p"].values)

wide.to_csv(SI / "table_pearson_r_pvalues.csv", float_format="%.4g")

# LaTeX tabular body --- one row per model, one cell per setting.
# Drop the trailing \\ on the last row so \bottomrule lands cleanly.
tex_lines = []
for model in MODEL_ORDER:
    row = wide.loc[MODEL_LABEL[model]]
    cells = [fmt_p(row[SETTING_LABEL[s]]) for s in SETTINGS_ORDER]
    tex_lines.append(f"{MODEL_LABEL[model]} & " + " & ".join(cells))
body = " \\\\\n".join(tex_lines) + "\n"

with open(SI / "table_pearson_r_pvalues.tex", "w", encoding="utf-8") as fh:
    fh.write(body)

print(f"Saved: {SI/'table_pearson_r_pvalues.csv'}")
print(f"Saved: {SI/'table_pearson_r_pvalues.tex'}")
print()
print(wide.to_string())

"""
Generate paper figures from experiment_results.csv.

Figures:
  fig1_main      — R^2 per model x setting (averaged across alloys+folds)
  fig2_per_alloy — Same broken down per target alloy (3-panel)
  fig3_leakage   — Filtered vs unfiltered side-by-side, highlighting Δ R^2
  fig_si_r_mae   — Pearson r and MAE companion (was fig4_r2_mae)
  fig6_gemini_zeroshot — Gemini in-context R^2 per setting vs zero-shot
  fig_si_*       — Per-fold scatter plots, error bars, etc.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Polished house style: serif fonts, despined axes, light dashed grid.
# Figure sizes are kept compact so the (absolute) fonts read larger.
POLISHED_RC = {
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "font.size": 12.5, "axes.labelsize": 12.5, "axes.titlesize": 12.5,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 10, "figure.titlesize": 12.5,
    "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.6,
}
plt.rcParams.update(POLISHED_RC)

# The five fitted models (Gemini handled separately as the in-context method).
FITTED = ["RF", "GBR", "MLP", "kNN_Tan", "ChemProp"]

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "experiment_results.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
SI_DIR = ROOT / "SI"
SI_DIR.mkdir(exist_ok=True)

SETTINGS_ORDER = ["exact", "close_filt", "close_unfilt", "far_filt", "far_unfilt"]
SETTING_LABELS = {
    "exact":         "Exact",
    "close_filt":    "Close\nfilt.",
    "close_unfilt":  "Close\nunfilt.",
    "far_filt":      "Far\nfilt.",
    "far_unfilt":    "Far\nunfilt.",
}
MODEL_ORDER = ["RF", "GBR", "MLP", "kNN_Tan", "ChemProp", "Gemini3.1Pro"]
MODEL_COLORS = {
    "RF":            "#d62728",
    "GBR":           "#ff7f0e",
    "MLP":           "#9467bd",
    "kNN_Tan":       "#2ca02c",
    "ChemProp":      "#1f77b4",
    "Gemini3.1Pro":  "#000000",
}
MODEL_LABELS = {
    "RF": "RF",
    "GBR": "GBR",
    "MLP": "MLP",
    "kNN_Tan": "kNN-Tanimoto",
    "ChemProp": "ChemProp (D-MPNN)",
    "Gemini3.1Pro": "Gemini 3.1 Pro",
}


def load():
    df = pd.read_csv(RESULTS)
    df = df.dropna(subset=["pearson_r"])
    return df


def fig_main_lineplot(df):
    """Two panels: R^2 (primary metric) and Pearson r (ranking) side by side.
    The r panel exposes the ranking-vs-calibration gap --- r stays high and
    positive even where R^2 collapses under molecule-disjoint filtering.
    Compact size (15% less wide / 25% less high than the prior single panel
    pair) with a top legend (3 cols x 2 rows)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.78))
    x = np.arange(len(SETTINGS_ORDER))
    specs = [("r2", r"$R^{2}$", (-1.5, 1.05), True),
             ("pearson_r", r"Pearson $r$", (0.0, 1.02), False)]
    for ax, (metric, ylab, ylim, zeroline) in zip(axes, specs):
        for model in MODEL_ORDER:
            sub = df[df["model"] == model]
            if len(sub) == 0:
                continue
            agg = sub.groupby("setting")[metric].agg(["mean", "std", "count"]).reindex(SETTINGS_ORDER)
            gem = (model == "Gemini3.1Pro")
            ax.errorbar(x, agg["mean"], yerr=agg["std"] / np.sqrt(agg["count"]),
                        marker="o", ms=6 if gem else 4.5, lw=2.6 if gem else 1.4,
                        alpha=1 if gem else 0.78, color=MODEL_COLORS[model],
                        label=MODEL_LABELS[model], capsize=2.5, zorder=6 if gem else 3)
        ax.set_xticks(x)
        ax.set_xticklabels([SETTING_LABELS[s] for s in SETTINGS_ORDER], fontsize=9)
        ax.set_xlabel("Transfer setting")
        ax.set_ylabel(ylab)
        if zeroline:
            ax.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3,
               frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_main.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_main.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_main.png/pdf")


def fig_per_alloy(df):
    """3 panels, one per target alloy. Compact size, top legend (3 cols x 2)."""
    alloys = sorted(df["alloy"].unique())
    fig, axes = plt.subplots(1, len(alloys), figsize=(7.9, 2.55), sharey=True)
    if len(alloys) == 1:
        axes = [axes]
    x = np.arange(len(SETTINGS_ORDER))
    for ax, alloy in zip(axes, alloys):
        sub_alloy = df[df["alloy"] == alloy]
        for model in MODEL_ORDER:
            sub = sub_alloy[sub_alloy["model"] == model]
            if len(sub) == 0:
                continue
            agg = sub.groupby("setting")["r2"].agg(["mean", "std", "count"]).reindex(SETTINGS_ORDER)
            gem = (model == "Gemini3.1Pro")
            ax.errorbar(x, agg["mean"], yerr=agg["std"] / np.sqrt(agg["count"]),
                        marker="o", ms=5 if gem else 4, lw=2.4 if gem else 1.3,
                        alpha=1 if gem else 0.75, color=MODEL_COLORS[model],
                        label=MODEL_LABELS[model], capsize=2, zorder=6 if gem else 3)
        ax.set_xticks(x)
        ax.set_xticklabels([SETTING_LABELS[s] for s in SETTINGS_ORDER], fontsize=9)
        ax.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax.grid(True, axis="y")
        ax.set_title(alloy, fontweight="bold")
        ax.set_ylim(-1.5, 1.05)
    axes[0].set_ylabel(r"$R^{2}$")
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3,
               frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_per_alloy.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_per_alloy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_per_alloy.png/pdf")


def fig_leakage_bars(df):
    """Bar-pair plot: filtered vs unfiltered for close + far, per model.
    Polished style, 10% smaller figure than before."""
    import matplotlib.patches as mpatches
    fig, axes = plt.subplots(1, 2, figsize=(6.66, 2.97), sharey=True)
    for ax, scope in zip(axes, ["close", "far"]):
        models = MODEL_ORDER
        x = np.arange(len(models))
        width = 0.38
        filt, unf, filt_se, unf_se = [], [], [], []
        for m in models:
            sub_f = df[(df["model"] == m) & (df["setting"] == f"{scope}_filt")]
            sub_u = df[(df["model"] == m) & (df["setting"] == f"{scope}_unfilt")]
            if len(sub_f) == 0 or len(sub_u) == 0:
                filt.append(np.nan); unf.append(np.nan)
                filt_se.append(0); unf_se.append(0); continue
            filt.append(sub_f["r2"].mean()); unf.append(sub_u["r2"].mean())
            filt_se.append(sub_f["r2"].std() / np.sqrt(len(sub_f)))
            unf_se.append(sub_u["r2"].std() / np.sqrt(len(sub_u)))

        cols = [MODEL_COLORS[m] for m in models]
        ax.bar(x - width/2, unf, width, yerr=unf_se, color=cols,
               alpha=0.95, capsize=2.5, label="Unfiltered")
        ax.bar(x + width/2, filt, width, yerr=filt_se, color=cols,
               alpha=0.42, capsize=2.5, hatch="//", label="Filtered")
        for i in range(len(models)):
            if not np.isnan(unf[i]) and not np.isnan(filt[i]):
                ax.text(i, 1.04, f"{unf[i]-filt[i]:+.2f}", ha="center",
                        va="bottom", fontsize=7.5, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models],
                           fontsize=8.5, rotation=30, ha="right")
        ax.set_xlabel(f"{scope.capitalize()} transfer", fontweight="bold")
        ax.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax.grid(axis="y")
        ax.set_ylim(-1.5, 1.15)

    axes[0].set_ylabel(r"$R^{2}$")
    handles = [mpatches.Patch(facecolor="0.5", label="Unfiltered"),
               mpatches.Patch(facecolor="0.5", alpha=0.42, hatch="//", label="Filtered")]
    axes[1].legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_leakage.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_leakage.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_leakage.png/pdf")


def fig_r_mae(df):
    """SI companion to fig1: Pearson r and MAE (R^2 is the main metric).
    Compact size, top legend (3 cols x 2 rows)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.6))
    x = np.arange(len(SETTINGS_ORDER))
    for metric, ax, ylab in [("pearson_r", axes[0], r"Pearson $r$"),
                             ("mae", axes[1], "MAE (% IE)")]:
        for model in MODEL_ORDER:
            sub = df[df["model"] == model]
            if len(sub) == 0:
                continue
            agg = sub.groupby("setting")[metric].agg(["mean", "std", "count"]).reindex(SETTINGS_ORDER)
            gem = (model == "Gemini3.1Pro")
            ax.errorbar(x, agg["mean"], yerr=agg["std"] / np.sqrt(agg["count"]),
                        marker="o", ms=5 if gem else 4, lw=2.4 if gem else 1.3,
                        alpha=1 if gem else 0.75, color=MODEL_COLORS[model],
                        label=MODEL_LABELS[model], capsize=2, zorder=6 if gem else 3)
        ax.set_xticks(x)
        ax.set_xticklabels([SETTING_LABELS[s] for s in SETTINGS_ORDER], fontsize=9)
        ax.set_ylabel(ylab)
        if metric == "pearson_r":
            ax.axhline(0, color="black", lw=0.6, alpha=0.5)
            ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3,
               frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_si_r_mae.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_si_r_mae.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_si_r_mae.png/pdf")


def summary_tables(df):
    """Save SI tables: per-cell mean ± std for r, R², MAE."""
    for metric in ["pearson_r", "r2", "mae"]:
        pivot = df.groupby(["model", "setting"])[metric].agg(["mean", "std"]).round(3)
        pivot.to_csv(SI_DIR / f"table_{metric}.csv")
        # Wide form
        wide_mean = df.groupby(["model", "setting"])[metric].mean().unstack().reindex(
            index=MODEL_ORDER, columns=SETTINGS_ORDER).round(3)
        wide_mean.to_csv(SI_DIR / f"table_{metric}_wide.csv")
    print("Saved SI tables to SI/")


def _load_gemini_zeroshot():
    """Compute per-alloy Pearson r/R2/MAE for zero-shot Gemini (no in-context
    examples). Returns dict alloy -> (r, R2, MAE) and the overall (r, R2, MAE)."""
    import json, glob
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score, mean_absolute_error
    out = {}
    all_y, all_p = [], []
    for f in sorted(glob.glob(str(ROOT / "results" / "gemini_zeroshot" / "*.json"))):
        d = json.load(open(f))
        ys, ps = [], []
        for mol, plist in d["predictions"].items():
            cleaned = [p for p in plist if p is not None and isinstance(p, (int, float)) and np.isfinite(p)]
            if not cleaned:
                continue
            ys.append(d["ground_truth"][mol])
            ps.append(np.mean(cleaned))
        if not ys:
            continue
        ys = np.array(ys); ps = np.array(ps)
        r, _ = pearsonr(ys, ps)
        out[d["alloy"]] = (r, r2_score(ys, ps), mean_absolute_error(ys, ps))
        all_y.extend(ys.tolist()); all_p.extend(ps.tolist())
    if all_y:
        ay = np.array(all_y); ap = np.array(all_p)
        r, _ = pearsonr(ay, ap)
        out["__all__"] = (r, r2_score(ay, ap), mean_absolute_error(ay, ap))
    return out


def fig_zeroshot_contamination(df):
    """Compare in-context Gemini (per setting) vs zero-shot Gemini, per alloy.

    Establishes that the in-context filtered performance is NOT explained by
    pre-training memorization of (molecule, IE) pairs.
    """
    zs = _load_gemini_zeroshot()
    if not zs:
        print("No zero-shot data; skipping fig_zeroshot.")
        return
    alloys = sorted([a for a in zs.keys() if a != "__all__"])
    settings = SETTINGS_ORDER

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    n_settings = len(settings)
    n_groups = len(alloys)
    width = 0.13
    x = np.arange(n_groups)

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_settings))
    for s_idx, setting in enumerate(settings):
        means = []
        for a in alloys:
            sub = df[(df["alloy"] == a) & (df["setting"] == setting) &
                     (df["model"] == "Gemini3.1Pro")]
            means.append(sub["r2"].mean() if len(sub) else np.nan)
        ax.bar(x + (s_idx - (n_settings)/2) * width, means, width,
               color=cmap[s_idx], label=SETTING_LABELS[setting].replace("\n", " "),
               edgecolor="black", linewidth=0.4)

    # Zero-shot bars use R^2 (index 1 in the (r, R^2, MAE) tuple)
    zs_vals = [zs[a][1] for a in alloys]
    ax.bar(x + (n_settings - (n_settings)/2) * width, zs_vals, width,
           color="white", edgecolor="black", hatch="///",
           label="Zero-shot")

    ax.set_xticks(x)
    ax.set_xticklabels(alloys)
    ax.set_ylabel(r"$R^{2}$")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.grid(axis="y")
    ax.set_ylim(-1.0, 0.9)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.95)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_gemini_zeroshot.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_gemini_zeroshot.pdf", bbox_inches="tight")
    plt.close(fig)

    # Also save a small CSV for the SI
    import csv
    with open(SI_DIR / "table_gemini_zeroshot.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alloy", "pearson_r", "R2", "MAE"])
        for a in alloys:
            r, r2, mae = zs[a]
            w.writerow([a, f"{r:.3f}", f"{r2:.3f}", f"{mae:.1f}"])
        if "__all__" in zs:
            r, r2, mae = zs["__all__"]
            w.writerow(["pooled", f"{r:.3f}", f"{r2:.3f}", f"{mae:.1f}"])
    print("Saved fig6_gemini_zeroshot.png/pdf and SI/table_gemini_zeroshot.csv")


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} rows")
    print(f"Coverage:\n{df.groupby(['model', 'setting']).size().unstack(fill_value=0)}")
    fig_main_lineplot(df)
    fig_per_alloy(df)
    fig_leakage_bars(df)
    fig_r_mae(df)
    fig_zeroshot_contamination(df)
    summary_tables(df)

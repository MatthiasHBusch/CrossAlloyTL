"""
Sample-count statistics for the dataset, target alloys, and pools.

Outputs:
  results/sample_counts.csv         — full per-alloy breakdown
  figures/fig0_sample_counts.png    — bar chart (target alloys highlighted)
  SI/table_sample_counts.csv        — paper-ready
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
})

from data_loader import (
    load_dataset, get_pools, prepare_alloy_data, canonical_smiles,
    TARGET_ALLOYS, K_FOLD,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
SI = ROOT / "SI"
for d in (RES, FIG, SI):
    d.mkdir(exist_ok=True)


def main():
    df = load_dataset()
    target_full, close_pool, far_pool = get_pools(df)

    # Alloy-level breakdown
    rows = []
    for alloy, sub in df.groupby("Alloy", dropna=True):
        n_records = len(sub)
        n_unique_mols = sub["canon_smi"].nunique()
        rows.append({
            "alloy": alloy,
            "base_material": sub["BaseMaterial"].iloc[0],
            "n_records": n_records,
            "n_unique_molecules": n_unique_mols,
            "is_target": alloy in TARGET_ALLOYS,
            "is_close_pool": (alloy not in TARGET_ALLOYS
                              and sub["BaseMaterial"].iloc[0] == "Mg"),
            "is_far_pool": sub["BaseMaterial"].iloc[0] != "Mg",
        })
    counts = pd.DataFrame(rows).sort_values(
        ["base_material", "n_records"], ascending=[True, False])
    counts.to_csv(RES / "sample_counts.csv", index=False)

    # Pool-level summary
    print(f"Total records:          {len(df)}")
    print(f"Total unique molecules: {df['canon_smi'].nunique()}")
    print()
    print("=== Target alloys (3) ===")
    for alloy in TARGET_ALLOYS:
        sub = df[df["Alloy"] == alloy]
        n_records = len(sub)
        n_mols = sub["canon_smi"].nunique()
        # Effective fold sample size after the largest-group filter
        df_alloy, _ = prepare_alloy_data(target_full, alloy)
        n_used = len(df_alloy)
        n_used_mols = df_alloy["canon_smi"].nunique()
        print(f"  {alloy:6s}: {n_records:4d} total records, "
              f"{n_mols:4d} unique mols. "
              f"Used in 5-fold: {n_used} records ({n_used_mols} mols)")

    print()
    print(f"=== Close pool (other Mg alloys) ===")
    print(f"  records: {len(close_pool)},  unique mols: {close_pool['canon_smi'].nunique()}")
    for alloy, sub in close_pool.groupby("Alloy"):
        print(f"    {alloy:18s}: {len(sub):4d} records, {sub['canon_smi'].nunique():3d} mols")

    print()
    print(f"=== Far pool (non-Mg base) ===")
    print(f"  records: {len(far_pool)},  unique mols: {far_pool['canon_smi'].nunique()}")
    for base, sub in far_pool.groupby("BaseMaterial"):
        print(f"    {base:8s}: {len(sub):5d} records, {sub['canon_smi'].nunique():4d} mols")

    # Test molecule overlap with pools (per fold)
    print()
    print(f"=== Test-fold molecule overlap with transfer pools ===")
    overlap_rows = []
    for alloy in TARGET_ALLOYS:
        df_alloy, fold_indices = prepare_alloy_data(target_full, alloy)
        for k, (_, te_idx) in enumerate(fold_indices):
            test_mols = set(df_alloy.iloc[te_idx]["canon_smi"].unique())
            close_mols = set(close_pool["canon_smi"].unique())
            far_mols = set(far_pool["canon_smi"].unique())
            n_in_close = len(test_mols & close_mols)
            n_in_far = len(test_mols & far_mols)
            n_close_records_filtered = (close_pool["canon_smi"]
                                         .isin(test_mols).sum())
            n_far_records_filtered = (far_pool["canon_smi"]
                                       .isin(test_mols).sum())
            overlap_rows.append({
                "alloy": alloy, "fold": k,
                "n_test_mols": len(test_mols),
                "n_test_in_close_pool": n_in_close,
                "n_test_in_far_pool": n_in_far,
                "n_close_records_removed_when_filtering": n_close_records_filtered,
                "n_far_records_removed_when_filtering": n_far_records_filtered,
            })
            print(f"  {alloy} fold={k}: test mols={len(test_mols):2d}, "
                  f"in close={n_in_close:2d} (-{n_close_records_filtered:3d} rec), "
                  f"in far={n_in_far:2d} (-{n_far_records_filtered:3d} rec)")

    pd.DataFrame(overlap_rows).to_csv(RES / "test_pool_overlap.csv", index=False)

    # ── Bar plot: only show alloys with >=10 records ──
    counts_main = counts[counts["n_records"] >= 10].copy()
    counts_sorted = counts_main.sort_values("n_records", ascending=True)
    fig, ax = plt.subplots(figsize=(5.3, 4.5))
    colors = []
    for _, row in counts_sorted.iterrows():
        if row["alloy"] in TARGET_ALLOYS:
            colors.append("#C0392B")
        elif row["base_material"] == "Mg":
            colors.append("#2471A3")
        else:
            colors.append("#7F8C8D")

    y = np.arange(len(counts_sorted))
    ax.barh(y, counts_sorted["n_records"], color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    labels = [f"{r['alloy']}" if r["base_material"] == "Mg"
              else f"{r['alloy']} ({r['base_material']})"
              for _, r in counts_sorted.iterrows()]
    ax.set_yticklabels(labels)
    ax.set_xlabel("Records in ExCorr")
    for i, (_, row) in enumerate(counts_sorted.iterrows()):
        ax.text(row["n_records"] + 10, i,
                f"{row['n_records']} ({row['n_unique_molecules']})",
                va="center", fontsize=9, color="black")

    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color="#C0392B",
                       label=f"Target alloys ({len(TARGET_ALLOYS)})"),
        mpatches.Patch(color="#2471A3", label="Close pool (other Mg)"),
        mpatches.Patch(color="#7F8C8D", label="Far pool (non-Mg)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0, counts_sorted["n_records"].max() * 1.30)
    plt.tight_layout()
    fig.savefig(FIG / "fig0_sample_counts.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig0_sample_counts.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {FIG/'fig0_sample_counts.png'}")

    # ── SI table ──
    si_table = counts[["alloy", "base_material", "n_records",
                       "n_unique_molecules", "is_target"]].copy()
    si_table.columns = ["Alloy", "Base material", "Records",
                        "Unique molecules", "Target alloy?"]
    si_table.to_csv(SI / "table_sample_counts.csv", index=False)
    print(f"Saved SI table to {SI/'table_sample_counts.csv'}")


if __name__ == "__main__":
    main()

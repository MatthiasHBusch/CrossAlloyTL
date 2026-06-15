"""Exact dataset/pool counts for the paper, from ExCorrDatasetClean.csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_dataset, get_pools, TARGET_ALLOYS

df = load_dataset()
print(f"Total records: {len(df)}")
print(f"Total unique canonical molecules: {df['canon_smi'].nunique()}")
print(f"Base materials: {sorted(df['BaseMaterial'].dropna().unique())}")
print(f"N alloys total (alloy one-hot dim): {df['Alloy'].dropna().nunique()}")

mg = df[df["BaseMaterial"] == "Mg"]
print(f"\nMg records: {len(mg)}, unique mols: {mg['canon_smi'].nunique()}")
print(f"Mg 'alloys' incl. tiny groups: {sorted(mg['Alloy'].dropna().unique())}")
print("Mg per-alloy counts:")
print(mg.groupby("Alloy").agg(records=("IE", "size"),
                              mols=("canon_smi", "nunique")).sort_values("records", ascending=False))

target, close, far = get_pools(df)
print(f"\nTarget pool (AZ31/AZ91/WE43): {len(target)} records, "
      f"{target['canon_smi'].nunique()} unique mols")
print(f"Close pool (all other Mg incl MgCa+misc): {len(close)} records, "
      f"{close['canon_smi'].nunique()} unique mols")
mgca = close[close["Alloy"] == "MgCa"]
print(f"  of which MgCa: {len(mgca)} records, {mgca['canon_smi'].nunique()} mols")
misc = close[close["Alloy"].isin(["MgAlZn", "AZ91D", "CPMgCa0.45Mn0.01"])]
print(f"  of which misc tiny groups: {len(misc)} records")
core = close[~close["Alloy"].isin(["MgCa", "MgAlZn", "AZ91D", "CPMgCa0.45Mn0.01"])]
print(f"  core 7 alloys: {len(core)} records, {core['canon_smi'].nunique()} unique mols")
print(f"Far pool (non-Mg): {len(far)} records, {far['canon_smi'].nunique()} unique mols")
print(f"Far base materials: {sorted(far['BaseMaterial'].dropna().unique())}")
print(f"\nCheck sum: target {len(target)} + close {len(close)} + far {len(far)} = "
      f"{len(target)+len(close)+len(far)}")

# Sanity: n_train for AZ31 fold 0 close_unfilt should be 60 + len(close)
print(f"\nExpected n_train close_unfilt = 60 + {len(close)} = {60+len(close)}")

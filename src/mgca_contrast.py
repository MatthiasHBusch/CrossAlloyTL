"""
MgCa CONTRAST experiment — same protocol as main runner, but for MgCa.

MgCa is the alloy that DOES NOT belong to the SAR-similarity cluster
(see Dir 11 paper, Spearman r = 0.16-0.35 with all other alloys).
We use it as a contrast: if the filtered-vs-unfiltered gap really
reflects molecule-level leakage (and not actual SAR transfer), then
MgCa should still see a collapse from unfiltered TL to filtered TL —
but the ABSOLUTE performance should be much lower, because the close
pool's molecules don't share MgCa's mechanism.

MgCa data is sparse (56 records, 49 unique molecules) so we use a
relaxed protocol:
  - take all 56 records
  - 4-fold CV (~14 train / ~14 test per fold)
  - same 5 transfer settings, same 5 models, same iterations

Results saved to results/mgca_contrast/<key>.json (one per combo).

Usage:
    C:/Users/mbusc/miniconda3/envs/xtb_env/python.exe mgca_contrast.py
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import (
    load_dataset, build_setting, build_features,
    SETTINGS,
)
from run_experiments import run_one
from models import (
    run_rf, run_gbr, run_mlp, run_knn_tanimoto, run_chemprop,
)

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results" / "mgca_contrast"
RES_DIR.mkdir(parents=True, exist_ok=True)

MGCA_K_FOLD = 4   # 56/4 = 14 per fold
MGCA_ALLOY = "MgCa"


def prepare_mgca(df):
    mgca = df[df["Alloy"] == MGCA_ALLOY].copy().reset_index(drop=True)
    n = len(mgca)
    fold_size = n // MGCA_K_FOLD
    fold_indices = []
    for k in range(MGCA_K_FOLD):
        test_start = fold_size * k
        test_end = fold_size * (k + 1) if k < MGCA_K_FOLD - 1 else n
        test_idx = list(range(test_start, test_end))
        train_idx = [i for i in range(n) if i not in test_idx]
        fold_indices.append((train_idx, test_idx))
    return mgca, fold_indices


def get_mgca_pools(df):
    """For MgCa contrast: close pool is all OTHER Mg alloys (incl. AZ31/91, WE43);
    far pool is all non-Mg base materials."""
    mg = df[df["BaseMaterial"] == "Mg"].copy()
    close = mg[mg["Alloy"] != MGCA_ALLOY].copy()
    far = df[df["BaseMaterial"] != "Mg"].copy()
    return close, far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-chemprop", action="store_true")
    ap.add_argument("--n-iter", type=int, default=25)
    ap.add_argument("--n-iter-chemprop", type=int, default=15)
    args = ap.parse_args()

    df = load_dataset()
    mgca_df, fold_indices = prepare_mgca(df)
    close_pool, far_pool = get_mgca_pools(df)

    alloy_list = sorted(df["Alloy"].dropna().unique().tolist())
    base_list = sorted(df["BaseMaterial"].dropna().unique().tolist())

    print(f"MgCa records: {len(mgca_df)}")
    print(f"MgCa unique mols: {mgca_df['canon_smi'].nunique()}")
    print(f"Folds: {MGCA_K_FOLD}")
    print(f"Close pool: {len(close_pool)}  Far pool: {len(far_pool)}")
    print()

    models = ["RF", "GBR", "MLP", "kNN_Tan"]
    if not args.no_chemprop:
        models.append("ChemProp")

    rows = []
    for fold_idx, (tr_idx, te_idx) in enumerate(fold_indices):
        target_train = mgca_df.iloc[tr_idx].copy()
        test_df = mgca_df.iloc[te_idx].copy()
        for setting in SETTINGS:
            for model in models:
                key = f"MgCa_f{fold_idx}_{setting}_{model}"
                out_file = RES_DIR / f"{key}.json"
                if out_file.exists():
                    continue
                t0 = time.time()
                print(f"  {key:55s}", end=" ", flush=True)
                try:
                    result = run_one(
                        MGCA_ALLOY, fold_idx, setting, model,
                        target_train, test_df, close_pool, far_pool,
                        alloy_list, base_list,
                        n_iter_classic=args.n_iter,
                        n_iter_chemprop=args.n_iter_chemprop)
                    if result is None:
                        print("[skipped]")
                        continue
                    rows.append(result)
                    with open(out_file, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"r2={result['r2']:+.3f}  r={result['pearson_r']:+.3f}  ({time.time()-t0:.0f}s)")
                except Exception as e:
                    print(f"FAIL: {e}")

    if rows:
        df_res = pd.DataFrame(rows)
        df_res.to_csv(ROOT / "results" / "mgca_contrast.csv", index=False)
        print(f"\nSaved {len(df_res)} rows.")


if __name__ == "__main__":
    main()

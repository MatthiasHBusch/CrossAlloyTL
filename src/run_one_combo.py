"""
Cluster-side wrapper: runs ONE (alloy, fold, setting, model) combo
and writes the result to a JSON file.

Usage:
    python run_one_combo.py --alloy AZ31 --fold 0 --setting exact --model RF
                            --out /path/to/result.json
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# Override DATA_PATH for cluster-local execution if needed
import data_loader as dl
from pathlib import Path

# On the cluster the data file lives next to the source
local_data = Path(__file__).resolve().parent / "ExCorrDatasetClean.csv"
if local_data.exists():
    dl.DATA_PATH = local_data

from data_loader import (
    load_dataset, get_pools, prepare_alloy_data,
    TARGET_ALLOYS, K_FOLD, SETTINGS,
)
from run_experiments import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloy", required=True, choices=TARGET_ALLOYS)
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument("--setting", required=True, choices=SETTINGS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-iter", type=int, default=25)
    ap.add_argument("--n-iter-chemprop", type=int, default=15)
    args = ap.parse_args()

    df = load_dataset()
    target_full, close_pool, far_pool = get_pools(df)
    alloy_list = sorted(df["Alloy"].dropna().unique().tolist())
    base_list = sorted(df["BaseMaterial"].dropna().unique().tolist())

    df_alloy, fold_indices = prepare_alloy_data(target_full, args.alloy)
    tr_idx, te_idx = fold_indices[args.fold]
    target_train = df_alloy.iloc[tr_idx].copy()
    test_df = df_alloy.iloc[te_idx].copy()

    print(f"Running {args.alloy} fold={args.fold} {args.setting} {args.model}",
          flush=True)
    t0 = time.time()
    result = run_one(
        args.alloy, args.fold, args.setting, args.model,
        target_train, test_df, close_pool, far_pool,
        alloy_list, base_list,
        n_iter_classic=args.n_iter,
        n_iter_chemprop=args.n_iter_chemprop)

    if result is None:
        print("Skipped (model unavailable).")
        sys.exit(0)

    print(f"Done in {time.time()-t0:.1f}s: r2={result['r2']:+.3f} r={result['pearson_r']:+.3f}",
          flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

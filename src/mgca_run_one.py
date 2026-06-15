"""Cluster wrapper for one MgCa contrast combo."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import data_loader as dl

# Cluster-local data path
local_data = Path(__file__).resolve().parent / "ExCorrDatasetClean.csv"
if local_data.exists():
    dl.DATA_PATH = local_data

from data_loader import load_dataset, SETTINGS
from run_experiments import run_one


MGCA_K_FOLD = 4
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--setting", required=True, choices=SETTINGS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-iter", type=int, default=25)
    ap.add_argument("--n-iter-chemprop", type=int, default=15)
    args = ap.parse_args()

    df = load_dataset()
    mgca_df, fold_indices = prepare_mgca(df)
    mg = df[df["BaseMaterial"] == "Mg"]
    close_pool = mg[mg["Alloy"] != MGCA_ALLOY].copy()
    far_pool = df[df["BaseMaterial"] != "Mg"].copy()

    alloy_list = sorted(df["Alloy"].dropna().unique().tolist())
    base_list = sorted(df["BaseMaterial"].dropna().unique().tolist())

    tr_idx, te_idx = fold_indices[args.fold]
    target_train = mgca_df.iloc[tr_idx].copy()
    test_df = mgca_df.iloc[te_idx].copy()

    print(f"Running MgCa fold={args.fold} {args.setting} {args.model}", flush=True)
    t0 = time.time()
    result = run_one(
        MGCA_ALLOY, args.fold, args.setting, args.model,
        target_train, test_df, close_pool, far_pool,
        alloy_list, base_list,
        n_iter_classic=args.n_iter,
        n_iter_chemprop=args.n_iter_chemprop)

    if result is None:
        print("Skipped (model unavailable).")
        return

    print(f"Done ({time.time()-t0:.1f}s): r2={result['r2']:+.3f} r={result['pearson_r']:+.3f}",
          flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

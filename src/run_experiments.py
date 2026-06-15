"""
Cross-alloy TL experiment runner.

For each (alloy, fold, setting, model) combination compute test predictions
and metrics (R2, MAE, Pearson r). Saves a flat row-wise CSV.

Settings: exact, close_unfilt, close_filt, far_unfilt, far_filt
Models:   RF, GBR, MLP, kNN_Tan, ChemProp
Iters:    25 per (RF/GBR/MLP/kNN_Tan), 15 for ChemProp
Folds:    5 per alloy
Alloys:   AZ31, AZ91, WE43

Total runs:
  3 alloys * 5 folds * 5 settings * 5 models = 375 model evaluations
  with 25/15 inner iterations each.

Usage:
    C:/Users/mbusc/miniconda3/envs/xtb_env/python.exe run_experiments.py
        [--alloy AZ31] [--setting exact] [--model RF] [--no-chemprop]
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from data_loader import (
    load_dataset, get_pools, prepare_alloy_data, build_setting,
    build_features, morgan_fp,
    TARGET_ALLOYS, K_FOLD, SETTINGS,
)
from models import (
    run_rf, run_gbr, run_mlp, run_knn_tanimoto, run_chemprop,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate(y_true, y_pred):
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return np.nan, np.nan, np.nan, np.nan
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r, p = pearsonr(y_true, y_pred)
    return r2, mae, r, p


def run_one(alloy, fold_idx, setting, model_name,
            target_train_df, test_df, close_pool, far_pool,
            alloy_list, base_list,
            n_iter_classic=25, n_iter_chemprop=15):
    """Run a single (alloy, fold, setting, model) combination."""
    train_df = build_setting(target_train_df, test_df, close_pool, far_pool, setting)
    feat = build_features(train_df, test_df, alloy_list, base_list)
    y_train = train_df["IE"].values.astype(float)
    y_test = test_df["IE"].values.astype(float)

    # Scale context
    sc_ctx = StandardScaler()
    ctx_tr_s = sc_ctx.fit_transform(feat["ctx_train"])
    ctx_te_s = sc_ctx.transform(feat["ctx_test"])

    # Build feature blocks per setting
    # Exact: alloy is single, no OHE needed
    # Close: alloy OHE makes sense (same base material)
    # Far:   alloy + base OHE makes sense (different base materials)
    if setting == "exact":
        X_train = np.hstack([feat["fp_train"], ctx_tr_s])
        X_test = np.hstack([feat["fp_test"], ctx_te_s])
    elif setting in ("close_unfilt", "close_filt"):
        X_train = np.hstack([feat["fp_train"], feat["alloy_train"], ctx_tr_s])
        X_test = np.hstack([feat["fp_test"], feat["alloy_test"], ctx_te_s])
    else:  # far_*
        X_train = np.hstack([feat["fp_train"], feat["alloy_train"],
                             feat["base_train"], ctx_tr_s])
        X_test = np.hstack([feat["fp_test"], feat["alloy_test"],
                            feat["base_test"], ctx_te_s])

    t0 = time.time()
    if model_name == "RF":
        y_pred = run_rf(X_train, y_train, X_test, n_iter=n_iter_classic)
    elif model_name == "GBR":
        y_pred = run_gbr(X_train, y_train, X_test, n_iter=n_iter_classic)
    elif model_name == "MLP":
        y_pred = run_mlp(X_train, y_train, X_test, n_iter=n_iter_classic)
    elif model_name == "kNN_Tan":
        # kNN works on FP only (Tanimoto over FP)
        y_pred = run_knn_tanimoto(feat["fp_train"], y_train,
                                  feat["fp_test"], n_iter=n_iter_classic, k=5)
    elif model_name == "ChemProp":
        # ChemProp uses raw SMILES, plus the same context as classical models
        if setting == "exact":
            x_d_train = ctx_tr_s
            x_d_test = ctx_te_s
        elif setting in ("close_unfilt", "close_filt"):
            x_d_train = np.hstack([feat["alloy_train"], ctx_tr_s])
            x_d_test = np.hstack([feat["alloy_test"], ctx_te_s])
        else:
            x_d_train = np.hstack([feat["alloy_train"], feat["base_train"],
                                   ctx_tr_s])
            x_d_test = np.hstack([feat["alloy_test"], feat["base_test"],
                                  ctx_te_s])
        y_pred = run_chemprop(train_df, test_df, x_d_train, x_d_test,
                              n_iter=n_iter_chemprop)
        if y_pred is None:
            return None
    else:
        raise ValueError(f"Unknown model: {model_name}")

    elapsed = time.time() - t0
    r2, mae, r, p = evaluate(y_test, y_pred)

    return {
        "alloy": alloy, "fold": fold_idx, "setting": setting,
        "model": model_name,
        "r2": r2, "mae": mae, "pearson_r": r, "pearson_p": p,
        "n_train": len(y_train), "n_test": len(y_test),
        "n_features": X_train.shape[1] if model_name != "ChemProp"
                       else (x_d_train.shape[1] if x_d_train is not None else 0),
        "elapsed_s": round(elapsed, 1),
        # Molecule-level predictions (needed for pooled significance tests).
        "y_test": [float(v) for v in y_test],
        "y_pred": [float(v) for v in y_pred],
        "test_smiles": test_df["canonical_SMILES"].tolist()
            if "canonical_SMILES" in test_df.columns
            else (test_df["isomeric_SMILES"].tolist()
                  if "isomeric_SMILES" in test_df.columns else []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloy", choices=TARGET_ALLOYS, default=None,
                    help="Run only this target alloy")
    ap.add_argument("--setting", choices=SETTINGS, default=None,
                    help="Run only this transfer setting")
    ap.add_argument("--model", default=None,
                    help="Run only this model (RF/GBR/MLP/kNN_Tan/ChemProp)")
    ap.add_argument("--no-chemprop", action="store_true",
                    help="Skip ChemProp (saves a lot of time)")
    ap.add_argument("--out", type=str,
                    default=str(RESULTS_DIR / "experiment_results.csv"))
    ap.add_argument("--n-iter", type=int, default=25)
    ap.add_argument("--n-iter-chemprop", type=int, default=15)
    args = ap.parse_args()

    df = load_dataset()
    target_full, close_pool, far_pool = get_pools(df)

    alloy_list = sorted(df["Alloy"].dropna().unique().tolist())
    base_list = sorted(df["BaseMaterial"].dropna().unique().tolist())

    print(f"Total records: {len(df)}")
    print(f"Target alloys: {TARGET_ALLOYS}")
    print(f"Close pool (other Mg): {len(close_pool)}")
    print(f"Far pool (non-Mg):     {len(far_pool)}")

    alloys = [args.alloy] if args.alloy else TARGET_ALLOYS
    settings = [args.setting] if args.setting else SETTINGS
    if args.model:
        models = [args.model]
    elif args.no_chemprop:
        models = ["RF", "GBR", "MLP", "kNN_Tan"]
    else:
        models = ["RF", "GBR", "MLP", "kNN_Tan", "ChemProp"]

    # Resume support: load existing results, skip already-done combos
    existing = pd.DataFrame()
    if Path(args.out).exists():
        existing = pd.read_csv(args.out)
        print(f"Resuming: {len(existing)} existing rows in {args.out}")

    def is_done(alloy, fold, setting, model):
        if existing.empty:
            return False
        m = ((existing["alloy"] == alloy) & (existing["fold"] == fold)
             & (existing["setting"] == setting) & (existing["model"] == model))
        return m.any()

    rows = existing.to_dict("records") if not existing.empty else []

    for alloy in alloys:
        df_alloy, fold_indices = prepare_alloy_data(target_full, alloy)
        for fold_idx, (tr_idx, te_idx) in enumerate(fold_indices):
            target_train = df_alloy.iloc[tr_idx].copy()
            test_df = df_alloy.iloc[te_idx].copy()

            for setting in settings:
                for model in models:
                    if is_done(alloy, fold_idx, setting, model):
                        continue
                    print(f"  {alloy} fold={fold_idx} {setting:13s} {model:9s}",
                          end=" ", flush=True)
                    try:
                        result = run_one(
                            alloy, fold_idx, setting, model,
                            target_train, test_df, close_pool, far_pool,
                            alloy_list, base_list,
                            n_iter_classic=args.n_iter,
                            n_iter_chemprop=args.n_iter_chemprop)
                        if result is None:
                            print("[skipped]")
                            continue
                        rows.append(result)
                        print(f"r2={result['r2']:+.3f}  r={result['pearson_r']:+.3f}"
                              f"  ({result['elapsed_s']:.0f}s)")
                        # Save after every model (resume safety)
                        pd.DataFrame(rows).to_csv(args.out, index=False)
                    except Exception as e:
                        print(f"FAIL: {e}")

    print(f"\nDone. {len(rows)} total rows saved to {args.out}")


if __name__ == "__main__":
    main()

"""
Rerun every (alloy, fold, setting, model) combination of the main experiment
and save the molecule-level y_test/y_pred so we can compute proper pooled
Pearson r + p across all 225 (mol, pred) pairs per (model, setting).

Output: results/predictions/<alloy>_f<fold>_<setting>_<model>.json
        with keys: alloy, fold, setting, model, y_test, y_pred, smiles, pearson_r, pearson_p

We monkey-patch run_one to capture the y_pred it computes; the metrics
remain identical to the original run_experiments.py output.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from data_loader import (load_dataset, get_pools, prepare_alloy_data,
                         TARGET_ALLOYS, K_FOLD, SETTINGS, build_setting,
                         build_features)
from models import (run_rf, run_gbr, run_mlp, run_knn_tanimoto, run_chemprop)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["RF", "GBR", "MLP", "kNN_Tan", "ChemProp"]


def run_and_capture(alloy, fold_idx, setting, model_name,
                    target_train_df, test_df, close_pool, far_pool,
                    alloy_list, base_list,
                    n_iter_classic=25, n_iter_chemprop=15):
    train_df = build_setting(target_train_df, test_df, close_pool, far_pool, setting)
    feat = build_features(train_df, test_df, alloy_list, base_list)
    y_train = train_df["IE"].values.astype(float)
    y_test = test_df["IE"].values.astype(float)
    sc_ctx = StandardScaler()
    ctx_tr_s = sc_ctx.fit_transform(feat["ctx_train"])
    ctx_te_s = sc_ctx.transform(feat["ctx_test"])
    if setting == "exact":
        X_train = np.hstack([feat["fp_train"], ctx_tr_s])
        X_test = np.hstack([feat["fp_test"], ctx_te_s])
    elif setting in ("close_unfilt", "close_filt"):
        X_train = np.hstack([feat["fp_train"], feat["alloy_train"], ctx_tr_s])
        X_test = np.hstack([feat["fp_test"], feat["alloy_test"], ctx_te_s])
    else:
        X_train = np.hstack([feat["fp_train"], feat["alloy_train"],
                             feat["base_train"], ctx_tr_s])
        X_test = np.hstack([feat["fp_test"], feat["alloy_test"],
                            feat["base_test"], ctx_te_s])

    if model_name == "RF":
        y_pred = run_rf(X_train, y_train, X_test, n_iter=n_iter_classic)
    elif model_name == "GBR":
        y_pred = run_gbr(X_train, y_train, X_test, n_iter=n_iter_classic)
    elif model_name == "MLP":
        y_pred = run_mlp(X_train, y_train, X_test, n_iter=n_iter_classic)
    elif model_name == "kNN_Tan":
        y_pred = run_knn_tanimoto(feat["fp_train"], y_train,
                                  feat["fp_test"], n_iter=n_iter_classic, k=5)
    elif model_name == "ChemProp":
        if setting == "exact":
            x_d_train = ctx_tr_s; x_d_test = ctx_te_s
        elif setting in ("close_unfilt", "close_filt"):
            x_d_train = np.hstack([feat["alloy_train"], ctx_tr_s])
            x_d_test = np.hstack([feat["alloy_test"], ctx_te_s])
        else:
            x_d_train = np.hstack([feat["alloy_train"], feat["base_train"], ctx_tr_s])
            x_d_test = np.hstack([feat["alloy_test"], feat["base_test"], ctx_te_s])
        y_pred = run_chemprop(train_df, test_df, x_d_train, x_d_test,
                              n_iter=n_iter_chemprop)
        if y_pred is None:
            return None
    return y_test, np.asarray(y_pred, dtype=float)


def main():
    df = load_dataset()
    target_full, close_pool, far_pool = get_pools(df)
    alloy_list = sorted(df["Alloy"].dropna().unique().tolist())
    base_list = sorted(df["BaseMaterial"].dropna().unique().tolist())

    total = len(TARGET_ALLOYS) * K_FOLD * len(SETTINGS) * len(MODELS)
    done = 0
    skipped = 0
    t_global = time.time()

    for alloy in TARGET_ALLOYS:
        df_alloy, fold_indices = prepare_alloy_data(target_full, alloy)
        for fold_idx, (tr_idx, te_idx) in enumerate(fold_indices):
            target_train = df_alloy.iloc[tr_idx].copy()
            test_df = df_alloy.iloc[te_idx].copy()
            smiles_test = test_df["canonical_SMILES"].tolist() \
                if "canonical_SMILES" in test_df.columns else \
                test_df.get("isomeric_SMILES", pd.Series([""] * len(test_df))).tolist()
            for setting in SETTINGS:
                for model in MODELS:
                    out = OUT_DIR / f"{alloy}_f{fold_idx}_{setting}_{model}.json"
                    if out.exists():
                        skipped += 1
                        done += 1
                        continue
                    print(f"[{done+1}/{total}] {alloy} f{fold_idx} {setting} {model} ...",
                          flush=True)
                    t0 = time.time()
                    try:
                        result = run_and_capture(
                            alloy, fold_idx, setting, model,
                            target_train, test_df, close_pool, far_pool,
                            alloy_list, base_list,
                            n_iter_classic=25, n_iter_chemprop=15)
                    except Exception as e:
                        print(f"   FAILED: {e}", flush=True)
                        done += 1
                        continue
                    if result is None:
                        done += 1
                        continue
                    y_test, y_pred = result
                    r, p = pearsonr(y_test, y_pred)
                    out_data = {
                        "alloy": alloy, "fold": fold_idx,
                        "setting": setting, "model": model,
                        "y_test": [float(v) for v in y_test],
                        "y_pred": [float(v) for v in y_pred],
                        "smiles": smiles_test,
                        "pearson_r": float(r),
                        "pearson_p": float(p),
                    }
                    with open(out, "w") as fh:
                        json.dump(out_data, fh)
                    done += 1
                    print(f"   done in {time.time()-t0:.1f}s  r={r:+.3f} p={p:.2e}",
                          flush=True)

    print(f"\nFinished {done} runs ({skipped} skipped, already on disk).")
    print(f"Total wall time: {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    main()

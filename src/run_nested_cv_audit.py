"""Nested-CV HP audit for dir-11 TL paper.

Question: do dir-11's transferability claims (close_filt r ≈ 0.30, close_unfilt
r ≈ 0.86 leakage demo) survive proper nested-CV HP tuning? Or are they (like
the 34-paper +0.255 lift) RF-defaults artefacts?

Protocol:
  - 5-fold outer × 3-fold inner CV per (alloy, setting, model)
  - HP search grids: RF (max_features, max_depth, min_samples_leaf),
    GBR (max_depth, learning_rate, min_samples_leaf), Ridge (α via RidgeCV).
  - n_iter=5 outer averaging for tree models.
  - Settings: exact, close_unfilt, close_filt, far_unfilt, far_filt
  - Models: rf, gbr, ridge

Output: results/nested_cv_audit_summary.csv with per-(setting, model)
        pooled r + jackknife 95% CI.
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, t as student_t
import warnings; warnings.filterwarnings("ignore")

from data_loader import (
    load_dataset, get_pools, prepare_alloy_data, build_setting,
    build_features, TARGET_ALLOYS, K_FOLD,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── HP grids ─────────────────────────────────────────────────────────
RF_GRID = [
    ("sqrt", None, 1), ("sqrt", None, 5), ("sqrt", 10, 5),
    (0.3, None, 1),    (0.3, None, 5),    (0.3, 10, 5),
    (0.5, None, 1),    (0.5, None, 5),
    (1.0, None, 1),    (1.0, None, 5),    (1.0, 10, 5),  # 1.0,1 = sklearn defaults
]
GBR_GRID = [
    # (max_depth, learning_rate, min_samples_leaf)
    (3, 0.05, 1), (3, 0.05, 5),
    (5, 0.05, 1), (5, 0.05, 5),  # 5,0.05,1 ≈ dir-11 paper config
    (5, 0.10, 1), (5, 0.10, 5),
    (7, 0.05, 5),
]
RIDGE_ALPHAS = np.logspace(-2, 3, 20)


def fisher_pool(rs, ns):
    rs = np.clip(np.asarray(rs, float), -0.999, 0.999)
    ns = np.asarray(ns, float)
    z = np.arctanh(rs)
    w = np.maximum(ns - 3, 0.1)
    return float(np.tanh((w * z).sum() / w.sum()))


def jackknife_ci(rs, ns, alpha=0.05):
    k = len(rs)
    if k < 2: return (np.nan, np.nan)
    rs, ns = np.asarray(rs), np.asarray(ns)
    full = fisher_pool(rs.tolist(), ns.tolist())
    leave_one = []
    for i in range(k):
        mask = np.arange(k) != i
        leave_one.append(fisher_pool(rs[mask].tolist(), ns[mask].tolist()))
    leave_one = np.asarray(leave_one)
    se = np.sqrt((k - 1) / k * np.sum((leave_one - leave_one.mean()) ** 2))
    crit = student_t.ppf(1 - alpha / 2, df=k - 1)
    return (full - crit * se, full + crit * se)


def inner_cv_score(model_fn, X_tr, y_tr, n_splits=3):
    rs = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for inner_tr, inner_va in kf.split(X_tr):
        try:
            m = model_fn()
            m.fit(X_tr[inner_tr], y_tr[inner_tr])
            pred = m.predict(X_tr[inner_va])
            r, _ = pearsonr(y_tr[inner_va], pred)
            if np.isfinite(r): rs.append(r)
        except Exception: continue
    return float(np.mean(rs)) if rs else -np.inf


def pick_rf_hp(X_tr, y_tr):
    best_hp, best_score = None, -np.inf
    for (mf, md, msl) in RF_GRID:
        def _make(seed=0, mf=mf, md=md, msl=msl):
            return RandomForestRegressor(
                n_estimators=200, max_features=mf, max_depth=md,
                min_samples_leaf=msl, random_state=seed, n_jobs=2)
        s = inner_cv_score(_make, X_tr, y_tr)
        if s > best_score: best_score, best_hp = s, (mf, md, msl)
    return best_hp, best_score


def pick_gbr_hp(X_tr, y_tr):
    best_hp, best_score = None, -np.inf
    for (md, lr, msl) in GBR_GRID:
        def _make(seed=0, md=md, lr=lr, msl=msl):
            return GradientBoostingRegressor(
                n_estimators=200, max_depth=md, learning_rate=lr,
                subsample=0.8, min_samples_leaf=msl, random_state=seed)
        s = inner_cv_score(_make, X_tr, y_tr)
        if s > best_score: best_score, best_hp = s, (md, lr, msl)
    return best_hp, best_score


def build_xy(train_df, test_df, alloy_list, base_list, setting):
    feat = build_features(train_df, test_df, alloy_list, base_list)
    sc_ctx = StandardScaler()
    ctx_tr = sc_ctx.fit_transform(feat["ctx_train"])
    ctx_te = sc_ctx.transform(feat["ctx_test"])
    if setting == "exact":
        Xtr = np.hstack([feat["fp_train"], ctx_tr])
        Xte = np.hstack([feat["fp_test"],  ctx_te])
    elif setting.startswith("close"):
        Xtr = np.hstack([feat["fp_train"], feat["alloy_train"], ctx_tr])
        Xte = np.hstack([feat["fp_test"],  feat["alloy_test"],  ctx_te])
    else:  # far_*
        Xtr = np.hstack([feat["fp_train"], feat["alloy_train"], feat["base_train"], ctx_tr])
        Xte = np.hstack([feat["fp_test"],  feat["alloy_test"],  feat["base_test"],  ctx_te])
    ytr = train_df["IE"].values.astype(float)
    yte = test_df["IE"].values.astype(float)
    return Xtr, Xte, ytr, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--settings", default="exact,close_unfilt,close_filt")
    ap.add_argument("--models", default="ridge,rf,gbr")
    args = ap.parse_args()

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"== Dir-11 nested-CV audit  settings={settings}  models={models} ==")
    df = load_dataset()
    target_full, close_pool, far_pool = get_pools(df)
    alloy_list = sorted(df["Alloy"].dropna().unique().tolist())
    base_list = sorted(df["BaseMaterial"].dropna().unique().tolist())

    fold_rows = []
    t_start = time.time()
    for alloy in TARGET_ALLOYS:
        df_alloy, outer_folds = prepare_alloy_data(target_full, alloy)
        for fi, (tr_idx, te_idx) in enumerate(outer_folds):
            outer_tr = df_alloy.iloc[tr_idx].copy()
            outer_te = df_alloy.iloc[te_idx].copy()
            for setting in settings:
                train_df = build_setting(outer_tr, outer_te, close_pool, far_pool, setting)
                Xtr, Xte, ytr, yte = build_xy(train_df, outer_te, alloy_list, base_list, setting)
                if len(ytr) < 30 or len(yte) < 5: continue

                for model in models:
                    t0 = time.time()
                    chosen_hp = {}
                    if model == "ridge":
                        m = RidgeCV(alphas=RIDGE_ALPHAS)
                        m.fit(Xtr, ytr)
                        chosen_hp["alpha"] = float(m.alpha_)
                        y_pred = m.predict(Xte)
                    elif model == "rf":
                        hp, _ = pick_rf_hp(Xtr, ytr)
                        chosen_hp.update({"max_features": hp[0], "max_depth": hp[1],
                                          "min_samples_leaf": hp[2]})
                        preds = []
                        for s in range(args.n_iter):
                            mm = RandomForestRegressor(
                                n_estimators=200, max_features=hp[0], max_depth=hp[1],
                                min_samples_leaf=hp[2], random_state=s, n_jobs=2)
                            mm.fit(Xtr, ytr); preds.append(mm.predict(Xte))
                        y_pred = np.mean(preds, axis=0)
                    elif model == "gbr":
                        hp, _ = pick_gbr_hp(Xtr, ytr)
                        chosen_hp.update({"max_depth": hp[0], "learning_rate": hp[1],
                                          "min_samples_leaf": hp[2]})
                        preds = []
                        for s in range(args.n_iter):
                            mm = GradientBoostingRegressor(
                                n_estimators=200, max_depth=hp[0],
                                learning_rate=hp[1], subsample=0.8,
                                min_samples_leaf=hp[2], random_state=s)
                            mm.fit(Xtr, ytr); preds.append(mm.predict(Xte))
                        y_pred = np.mean(preds, axis=0)
                    else: continue

                    r, _ = pearsonr(yte, y_pred)
                    if not np.isfinite(r): r = np.nan
                    elapsed = time.time() - t0
                    fold_rows.append({
                        "alloy": alloy, "outer_fold": fi, "setting": setting,
                        "model": model, "n_train": len(ytr), "n_test": len(yte),
                        "r": round(r, 4) if np.isfinite(r) else None,
                        "chosen_hp": chosen_hp, "elapsed_s": round(elapsed, 1),
                    })
                    print(f"  {alloy}/f{fi}/{setting:13s}/{model:5s}  r={r:+.3f}  hp={chosen_hp}  ({elapsed:.0f}s)")

    out_json = RESULTS_DIR / "nested_cv_audit_folds.json"
    with open(out_json, "w") as f:
        json.dump(fold_rows, f, indent=1, default=float)
    print(f"\nSaved {len(fold_rows)} fold rows → {out_json}")
    print(f"Elapsed: {(time.time()-t_start)/60:.1f} min")

    # Summary
    summary_rows = []
    for setting in settings:
        for model in models:
            per_alloy = {}
            for r in fold_rows:
                if r["setting"] != setting or r["model"] != model: continue
                per_alloy.setdefault(r["alloy"], []).append(r["r"])
            rs, ns = [], []; per_alloy_r = {}
            for alloy, alloy_rs in per_alloy.items():
                arr = [x for x in alloy_rs if x is not None and np.isfinite(x)]
                if len(arr) < 2: continue
                pool_r = fisher_pool(arr, [15] * len(arr))
                rs.append(pool_r); ns.append(15 * len(arr))
                per_alloy_r[alloy] = round(pool_r, 4)
            if not rs: continue
            cross_pool = fisher_pool(rs, ns)
            ci_lo, ci_hi = jackknife_ci(rs, ns)
            summary_rows.append({
                "setting": setting, "model": model,
                "pooled_r": round(cross_pool, 4),
                "ci_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
                "ci_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
                "n_alloys": len(rs),
                "per_alloy_r": json.dumps(per_alloy_r, sort_keys=True),
            })

    sdf = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "nested_cv_audit_summary.csv"
    sdf.to_csv(summary_path, index=False)
    print(f"Saved summary → {summary_path}")

    print("\n══ Nested-CV pooled r (per setting × model, jackknife 95% CI) ══")
    for setting in settings:
        for model in models:
            sub = sdf[(sdf.setting == setting) & (sdf.model == model)]
            if sub.empty: continue
            row = sub.iloc[0]
            ci = (f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]"
                  if row["ci_lo"] is not None else "[—]")
            print(f"  {setting:13s}/{model:5s}  r={row['pooled_r']:+.3f}  CI={ci}  "
                  f"per-alloy={row['per_alloy_r']}")


if __name__ == "__main__":
    main()

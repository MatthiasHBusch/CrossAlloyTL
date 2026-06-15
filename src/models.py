"""
Model registry for cross-alloy TL paper.

Five model families:
  1. RF        — Random Forest (FP only, ensemble of 200 trees)
  2. GBR       — Gradient Boosting Regression (FP only, 200 trees)
  3. MLP       — Multi-Layer Perceptron on FP (256-128-64)
  4. ChemProp  — D-MPNN graph neural network on SMILES
  5. kNN-Tan   — k-Nearest-Neighbour with Tanimoto-similarity kernel (k=5)

All models accept Morgan FP (2048 bits, radius 2). ChemProp uses raw SMILES
plus optional context features (alloy OHE + concentration).

Each `run_<model>` function returns a numpy array of test predictions,
averaged over `n_iter` random seeds.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings("ignore")


# ── Random Forest ────────────────────────────────────────────────────────────

def run_rf(X_train, y_train, X_test, n_iter=25):
    preds = []
    for seed in range(n_iter):
        rf = RandomForestRegressor(
            n_estimators=200, random_state=seed, n_jobs=2)
        rf.fit(X_train, y_train)
        preds.append(rf.predict(X_test))
    return np.mean(preds, axis=0)


# ── Gradient Boosting ────────────────────────────────────────────────────────

def run_gbr(X_train, y_train, X_test, n_iter=25):
    preds = []
    for seed in range(n_iter):
        gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=seed)
        gbr.fit(X_train, y_train)
        preds.append(gbr.predict(X_test))
    return np.mean(preds, axis=0)


# ── MLP ──────────────────────────────────────────────────────────────────────

def run_mlp(X_train, y_train, X_test, n_iter=25):
    sc_X = StandardScaler()
    X_tr_s = sc_X.fit_transform(X_train)
    X_te_s = sc_X.transform(X_test)
    sc_y = StandardScaler()
    y_tr_s = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    preds = []
    for seed in range(n_iter):
        mlp = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            solver="adam", learning_rate_init=1e-3, max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            batch_size=32, random_state=seed)
        mlp.fit(X_tr_s, y_tr_s)
        pred_s = mlp.predict(X_te_s)
        preds.append(sc_y.inverse_transform(pred_s.reshape(-1, 1)).ravel())
    return np.mean(preds, axis=0)


# ── kNN with Tanimoto similarity ─────────────────────────────────────────────

def tanimoto_distance(X_a, X_b):
    """Pairwise Tanimoto distance between binary fingerprints.
    distance = 1 - similarity = 1 - (a . b) / (|a|^2 + |b|^2 - a . b)
    """
    Xa = X_a.astype(bool).astype(np.float32)
    Xb = X_b.astype(bool).astype(np.float32)
    inter = Xa @ Xb.T
    sum_a = Xa.sum(axis=1, keepdims=True)
    sum_b = Xb.sum(axis=1, keepdims=True).T
    union = sum_a + sum_b - inter
    sim = np.where(union > 0, inter / union, 0.0)
    return 1.0 - sim


def run_knn_tanimoto(X_train, y_train, X_test, n_iter=25, k=5):
    """k-NN regression with Tanimoto distance; n_iter only varies tie-breaking
    via random subsampling, since the base method is deterministic."""
    # Pairwise Tanimoto distances test x train
    dist = tanimoto_distance(X_test, X_train)  # (n_test, n_train)
    rng = np.random.default_rng(0)
    preds_iter = []
    for seed in range(n_iter):
        # Tie-break by random perturbation
        rng_loc = np.random.default_rng(seed)
        pert = rng_loc.uniform(0, 1e-9, size=dist.shape)
        d_jitter = dist + pert
        # k nearest indices per test point
        idx = np.argpartition(d_jitter, kth=min(k, dist.shape[1]-1), axis=1)[:, :k]
        # Distance-weighted mean
        preds_test = np.zeros(len(X_test))
        for i in range(len(X_test)):
            neigh_idx = idx[i]
            neigh_d = dist[i, neigh_idx]
            # Inverse-distance weights (smoothed)
            w = 1.0 / (neigh_d + 0.05)
            preds_test[i] = np.sum(w * y_train[neigh_idx]) / w.sum()
        preds_iter.append(preds_test)
    return np.mean(preds_iter, axis=0)


# ── ChemProp wrapper ─────────────────────────────────────────────────────────

def run_chemprop(train_df, test_df, x_d_train, x_d_test, n_iter=15):
    """ChemProp D-MPNN with optional extra descriptors x_d."""
    try:
        import torch
        torch.set_num_threads(2)
        from chemprop import data as cpd, featurizers as cpf
        from chemprop import models as cpm, nn as cpnn
        from lightning import pytorch as pl
    except ImportError:
        return None

    n_xd = x_d_train.shape[1] if x_d_train is not None else 0
    if n_xd > 0:
        sc = StandardScaler()
        x_d_train_s = sc.fit_transform(x_d_train).astype(np.float32)
        x_d_test_s = sc.transform(x_d_test).astype(np.float32)
    else:
        x_d_train_s = x_d_test_s = None

    all_preds = []
    for it in range(n_iter):
        train_dps = []
        for i, (_, row) in enumerate(train_df.iterrows()):
            xd = (np.asarray(x_d_train_s[i], dtype=np.float32)
                  if x_d_train_s is not None else None)
            dp = cpd.MoleculeDatapoint.from_smi(
                smi=row["isomeric_SMILES"], y=[float(row["IE"])], x_d=xd)
            train_dps.append(dp)

        test_dps = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            xd = (np.asarray(x_d_test_s[i], dtype=np.float32)
                  if x_d_test_s is not None else None)
            dp = cpd.MoleculeDatapoint.from_smi(
                smi=row["isomeric_SMILES"], y=[float(row["IE"])], x_d=xd)
            test_dps.append(dp)

        feat = cpf.SimpleMoleculeMolGraphFeaturizer()
        train_dset = cpd.MoleculeDataset(train_dps, feat)
        tgt_scaler = train_dset.normalize_targets()
        test_dset = cpd.MoleculeDataset(test_dps, feat)
        test_dset.normalize_targets(tgt_scaler)

        train_loader = cpd.build_dataloader(train_dset, num_workers=0, shuffle=True)
        test_loader = cpd.build_dataloader(test_dset, num_workers=0, shuffle=False)

        mp = cpnn.BondMessagePassing()
        agg = cpnn.MeanAggregation()
        out_tf = cpnn.UnscaleTransform.from_standard_scaler(tgt_scaler)
        ffn = cpnn.RegressionFFN(
            input_dim=mp.output_dim + n_xd, hidden_dim=300,
            n_layers=5, activation="relu", output_transform=out_tf)
        mpnn = cpm.MPNN(mp, agg, ffn, True,
                        [cpnn.metrics.RMSE(), cpnn.metrics.MAE()])

        trainer = pl.Trainer(
            logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            accelerator="auto", devices=1, max_epochs=5)
        trainer.fit(mpnn, train_loader)

        preds_batches = trainer.predict(mpnn, test_loader)
        pred = np.concatenate([b.cpu().numpy().flatten() for b in preds_batches])
        all_preds.append(pred)

    return np.mean(all_preds, axis=0)


MODEL_REGISTRY = {
    "RF":       run_rf,
    "GBR":      run_gbr,
    "MLP":      run_mlp,
    "kNN_Tan":  run_knn_tanimoto,
    "ChemProp": run_chemprop,   # special — different signature
}

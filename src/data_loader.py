"""
Data loading for cross-alloy TL paper.

Mirrors the MgProjectTransferLearning protocol:
- 3 target alloys: AZ31, AZ91, WE43
- For each alloy: select the largest experimental group (same Method,
  AggressiveComponent, Operating_Concentration_mM), take first 75 records.
- 5-fold CV: 60 train / 15 test per fold.

Transfer pools:
- Close pool: all OTHER Mg-base alloys (~839 records).
- Far pool:  all NON-Mg base materials (~3000 records).

Filtered pools: same pools but with test-fold molecules (canonical SMILES)
removed to eliminate label-leakage via molecule reappearance.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ExCorrDatasetClean.csv"

TARGET_ALLOYS = ["AZ31", "AZ91", "WE43"]
K_FOLD = 5
N_BITS = 2048
RADIUS = 2


def parse_aggressive_component(text):
    hcl_perc, nacl_perc = 0.0, 0.0
    if pd.isna(text) or not isinstance(text, str):
        return hcl_perc, nacl_perc
    hcl_m = re.search(r'(\d+(?:[.,]\d+)?)\s*M\b\s*[,:;]?\s*HCl', text, re.I)
    hcl_p = re.search(r'(\d+(?:[.,]\d+)?)(?:\s*(?:wt\.?\s*%|%))?\s*HCl\b', text, re.I)
    hcl_mm = re.search(r'(\d+(?:[.,]\d+)?)\s*mM\s*HCl', text, re.I)
    if hcl_m:
        hcl_perc = float(hcl_m.group(1).replace(',', '.')) * 36.46 / 10
    elif hcl_p and not hcl_m:
        hcl_perc = float(hcl_p.group(1).replace(',', '.'))
    elif hcl_mm:
        hcl_perc = float(hcl_mm.group(1).replace(',', '.')) / 1000 * 36.46 / 10
    nacl_m = re.search(r'(\d+(?:[.,]\d+)?)\s*M\b\s*[,:;]?\s*NaCl', text, re.I)
    nacl_p = re.search(r'(\d+(?:[.,]\d+)?)(?:\s*(?:wt\.?\s*%|%))?\s*NaCl\b', text, re.I)
    nacl_mm = re.search(r'(\d+(?:[.,]\d+)?)\s*mM\s*NaCl', text, re.I)
    if nacl_m:
        nacl_perc = float(nacl_m.group(1).replace(',', '.')) * 58.44 / 10
    elif nacl_p and not nacl_m:
        nacl_perc = float(nacl_p.group(1).replace(',', '.'))
    elif nacl_mm:
        nacl_perc = float(nacl_mm.group(1).replace(',', '.')) / 1000 * 58.44 / 10
    return hcl_perc, nacl_perc


def morgan_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(N_BITS, dtype=np.float32)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(
        mol, RADIUS, N_BITS, useChirality=True), dtype=np.float32)


def canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smi


def load_dataset():
    """Load and prepare the full ExCorr dataset with parsed features."""
    df = pd.read_csv(DATA_PATH)
    parsed = df["AggressiveComponent"].apply(
        lambda x: pd.Series(parse_aggressive_component(x),
                            index=["HCl_pct", "NaCl_pct"]))
    df = pd.concat([df, parsed], axis=1)
    df["conc_mM"] = df["Operating_Concentration_mM"].fillna(0).astype(float)
    df["canon_smi"] = df["isomeric_SMILES"].apply(canonical_smiles)
    return df


def get_pools(df):
    """Split full dataset into target / close / far pools.
       close: Mg base, NOT in TARGET_ALLOYS.
       far:   non-Mg base.
    """
    mg = df[df["BaseMaterial"] == "Mg"].copy()
    target = mg[mg["Alloy"].isin(TARGET_ALLOYS)].copy()
    close = mg[~mg["Alloy"].isin(TARGET_ALLOYS)].copy()
    far = df[df["BaseMaterial"] != "Mg"].copy()
    return target, close, far


def prepare_alloy_data(mg, alloy, k_fold=K_FOLD):
    """Take target alloy, pick largest experimental group, top-75 rows, k-fold."""
    df_alloy = mg[mg["Alloy"] == alloy].copy()
    grouped = df_alloy.groupby(
        ["BaseMaterial", "Alloy", "Method", "AggressiveComponent",
         "Operating_Concentration_mM"])
    largest_group = grouped.size().idxmax()
    df_alloy = grouped.get_group(largest_group).copy()
    if "index" in df_alloy.columns:
        dup_idx = df_alloy["index"][df_alloy["index"].duplicated(keep=False)].unique()
        df_alloy = df_alloy[~df_alloy["index"].isin(dup_idx)]
    df_alloy = df_alloy.head(75).reset_index(drop=True)

    fold_indices = []
    fold_size = 75 // k_fold
    for k in range(k_fold):
        test_idx = list(range(fold_size * k, fold_size * (k + 1)))
        train_idx = [i for i in range(len(df_alloy)) if i not in test_idx]
        fold_indices.append((train_idx, test_idx))
    return df_alloy, fold_indices


# ── Setting builders ────────────────────────────────────────────────────────

def build_setting(target_train, test_df, close_pool, far_pool,
                  setting):
    """Construct the training set for a given transfer setting.

    Parameters
    ----------
    target_train : 60 rows of target alloy (the train fold)
    test_df      : 15 rows of target alloy (the held-out test fold)
    close_pool   : full close transfer pool (other Mg alloys)
    far_pool     : full far transfer pool (non-Mg base materials)
    setting      : one of {'exact', 'close_unfilt', 'close_filt',
                            'far_unfilt', 'far_filt'}

    Returns
    -------
    train_full : pd.DataFrame
    """
    test_canon = set(test_df["canon_smi"].unique())

    if setting == "exact":
        return target_train.copy()

    if setting == "close_unfilt":
        return pd.concat([target_train, close_pool], ignore_index=True)

    if setting == "close_filt":
        close_f = close_pool[~close_pool["canon_smi"].isin(test_canon)]
        return pd.concat([target_train, close_f], ignore_index=True)

    if setting == "far_unfilt":
        return pd.concat([target_train, close_pool, far_pool],
                         ignore_index=True)

    if setting == "far_filt":
        close_f = close_pool[~close_pool["canon_smi"].isin(test_canon)]
        far_f = far_pool[~far_pool["canon_smi"].isin(test_canon)]
        return pd.concat([target_train, close_f, far_f], ignore_index=True)

    raise ValueError(f"Unknown setting: {setting}")


def build_features(train_df, test_df, alloy_list, base_material_list):
    """Build (FP, context, alloy_OHE, base_OHE) feature blocks for train/test.

    Returns dict with keys:
      fp_train, fp_test:        (n, 2048) Morgan FP
      ctx_train, ctx_test:      (n, 3)    HCl%, NaCl%, conc
      alloy_train, alloy_test:  (n, |alloys|) one-hot
      base_train,  base_test:   (n, |bases|) one-hot for base material
    """
    fp_tr = np.vstack([morgan_fp(s) for s in train_df["isomeric_SMILES"]])
    fp_te = np.vstack([morgan_fp(s) for s in test_df["isomeric_SMILES"]])

    ctx_tr = train_df[["HCl_pct", "NaCl_pct", "conc_mM"]].values.astype(np.float32)
    ctx_te = test_df[["HCl_pct", "NaCl_pct", "conc_mM"]].values.astype(np.float32)

    def ohe(values, vocab):
        out = np.zeros((len(values), len(vocab)), dtype=np.float32)
        for i, v in enumerate(values):
            if v in vocab:
                out[i, vocab.index(v)] = 1.0
        return out

    alloy_tr = ohe(train_df["Alloy"].values, alloy_list)
    alloy_te = ohe(test_df["Alloy"].values, alloy_list)
    base_tr = ohe(train_df["BaseMaterial"].values, base_material_list)
    base_te = ohe(test_df["BaseMaterial"].values, base_material_list)

    return {
        "fp_train": fp_tr, "fp_test": fp_te,
        "ctx_train": ctx_tr, "ctx_test": ctx_te,
        "alloy_train": alloy_tr, "alloy_test": alloy_te,
        "base_train": base_tr, "base_test": base_te,
    }


SETTINGS = ["exact", "close_unfilt", "close_filt", "far_unfilt", "far_filt"]

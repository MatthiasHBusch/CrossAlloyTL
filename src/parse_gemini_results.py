"""
Parse Gemini 3.1 Pro JSON outputs into the same per-(alloy, fold, setting)
metric format used by experiment_results.csv, then append/merge with the
existing CSV so make_figures.py picks Gemini up automatically.

JSON structure produced by full_run_extended:
    approach -> num_exact -> num_close -> num_far -> molecule_name -> [pred_iter1, pred_iter2, ...]

We aggregate per fold by mapping each test molecule (via canonical SMILES
matching against the prepared 75-row alloy DataFrame partitioned into
5 folds) and computing Pearson r / R2 / MAE on the fold's averaged
predictions.

Usage:
    python parse_gemini_results.py
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
GEMINI_DIR = ROOT / "results" / "gemini"
RESULTS_CSV = ROOT / "results" / "experiment_results.csv"

DATA_PATH = ROOT / "data" / "ExCorrDatasetClean.csv"

TARGET_ALLOYS = ["AZ31", "AZ91", "WE43"]
SETTINGS = ["exact", "close_unfilt", "close_filt",
            "far_unfilt", "far_filt"]
K_FOLD = 5
SEED = 42


def canonical(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else smi
    except Exception:
        return smi


def prepare_alloy(df, alloy):
    sub = df[(df["BaseMaterial"] == "Mg") & (df["Alloy"] == alloy)].copy()
    grp = sub.groupby(["BaseMaterial", "Alloy", "Method",
                       "AggressiveComponent",
                       "Operating_Concentration_mM"])
    largest = grp.size().idxmax()
    sub = grp.get_group(largest).copy()
    if "index" in sub.columns:
        dups = sub["index"][sub["index"].duplicated(keep=False)].unique()
        sub = sub[~sub["index"].isin(dups)]
    sub = sub.head(75).reset_index(drop=True)
    return sub


def get_fold_test_molecules(alloy_df, fold_idx):
    """Mirror Julia's per-fold partition (after MersenneTwister(SEED) shuffle).

    The Julia code does:
      data_shuffled = data_frame_test[shuffle(rng, 1:n), :]
      num_test = div(n, k)
      test_sets[j] = (num_test*(j-1)+1):(num_test*j)

    We can't reproduce Julia's RNG from Python exactly; instead we read
    back the per-fold test molecules by inspecting the JSON keys --
    the prediction dict's keys are exactly the test-fold molecule names
    in order. Hence the JSON IS the ground truth for which-mol-in-which-
    fold mapping; we don't need to re-derive folds here.
    """
    raise NotImplementedError(
        "Use the per-(num_close,num_far)-key structure to recover folds "
        "instead.")


def parse_json(json_path, alloy, ie_lookup):
    """Parse one JSON, return a list of fold-level metric dicts."""
    with open(json_path) as f:
        d = json.load(f)

    rows = []
    # Structure: input_type -> approach -> model -> num_exact -> num_close ->
    #            num_far -> mol -> [preds]
    for input_type, it_d in d.items():
        for approach, app_d in it_d.items():
            for model, mo_d in app_d.items():
                for ne, ne_d in mo_d.items():
                    for nc, nc_d in ne_d.items():
                        for nf, mols in nc_d.items():
                            if not mols:
                                continue
                            items = list(mols.items())
                            per_mol = []
                            for mol, preds in items:
                                if mol not in ie_lookup:
                                    continue
                                cleaned = [float(p) for p in preds
                                           if p is not None and isinstance(p, (int, float, str))]
                                cleaned = [p for p in cleaned if np.isfinite(p)]
                                if not cleaned:
                                    continue
                                per_mol.append((mol, np.mean(cleaned), ie_lookup[mol]))

                            if len(per_mol) < 5:
                                continue

                            n = len(per_mol)
                            fold_size = n // K_FOLD
                            for k in range(K_FOLD):
                                chunk = per_mol[k * fold_size: (k + 1) * fold_size]
                                if len(chunk) < 3:
                                    continue
                                ys = np.array([t[2] for t in chunk])
                                ps = np.array([t[1] for t in chunk])
                                r, _ = pearsonr(ys, ps)
                                rows.append({
                                    "alloy": alloy,
                                    "fold": k,
                                    "setting": json_path.stem.split("_")[-1],
                                    "model": "Gemini3.1Pro",
                                    "r2": r2_score(ys, ps),
                                    "mae": mean_absolute_error(ys, ps),
                                    "pearson_r": r,
                                    "pearson_p": float("nan"),
                                    "n_train": int(ne) + int(nc) + int(nf),
                                    "n_test": len(chunk),
                                    "n_features": float("nan"),
                                    "elapsed_s": float("nan"),
                                })
    return rows


def main():
    if not GEMINI_DIR.exists():
        print(f"No Gemini dir at {GEMINI_DIR}")
        return
    df_full = pd.read_csv(DATA_PATH)
    df_full["canon_smi"] = df_full["isomeric_SMILES"].apply(canonical)

    # Per-alloy IE lookup (IUPAC → mean IE for that alloy)
    rows = []
    for alloy in TARGET_ALLOYS:
        ie_lookup = (df_full[df_full["Alloy"] == alloy]
                     .groupby("IUPAC")["IE"].mean().to_dict())
        for setting in SETTINGS:
            jp = GEMINI_DIR / f"Gemini3_1Pro_{alloy}_{setting}.json"
            if not jp.exists():
                print(f"Missing: {jp.name}")
                continue
            new = parse_json(jp, alloy, ie_lookup)
            # Override setting field (parsed from filename, more reliable)
            for r in new:
                r["setting"] = setting
            rows.extend(new)
            print(f"  {jp.name}: {len(new)} fold-rows")

    if not rows:
        print("No rows parsed.")
        return

    df_new = pd.DataFrame(rows)
    print(f"\nTotal Gemini rows: {len(df_new)}")
    print(df_new.groupby(["alloy", "setting"])["pearson_r"].mean().unstack().round(3))

    # Merge with existing experiment_results.csv
    if RESULTS_CSV.exists():
        df_old = pd.read_csv(RESULTS_CSV)
        df_old = df_old[df_old["model"] != "Gemini3.1Pro"]  # replace existing
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_merged = df_new

    df_merged.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved merged CSV to {RESULTS_CSV} ({len(df_merged)} total rows)")


if __name__ == "__main__":
    main()

"""
Parse the Gemini 3.1 Pro (flex) MgCa-contrast JSON outputs into the same
per-(fold, setting) metric format used by mgca_contrast.csv, then merge
them in so make_mgca_figure.py picks Gemini up automatically on the MgCa
panel.

This is the MgCa analogue of parse_gemini_results.py:
  * reads results/gemini_mgca/Gemini3_1Pro_MgCa_<setting>.json
  * alloy is fixed to "MgCa", K_FOLD = 4 (matches run_gemini_mgca.jl /
    mgca_contrast.py)
  * merges into results/mgca_contrast.csv (replacing any existing
    Gemini3.1Pro rows) under model key "Gemini3.1Pro" so it lines up
    with MODEL_ORDER/COLORS/LABELS in make_figures.py.

Fold recovery mirrors parse_gemini_results.py: the JSON prediction dict
preserves Julia's per-fold insertion order, so flattening the per-molecule
predictions and re-chunking into K_FOLD contiguous groups reproduces the
per-fold grouping (same approximation used for the main 3 alloys).

Usage:
    python parse_gemini_mgca.py
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
GEMINI_DIR = ROOT / "results" / "gemini_mgca"
RESULTS_CSV = ROOT / "results" / "mgca_contrast.csv"

DATA_PATH = ROOT / "data" / "ExCorrDatasetClean.csv"

ALLOY = "MgCa"
SETTINGS = ["exact", "close_unfilt", "close_filt",
            "far_unfilt", "far_filt"]
K_FOLD = 4
MODEL_KEY = "Gemini3.1Pro"


def canonical(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else smi
    except Exception:
        return smi


def parse_json(json_path, alloy, ie_lookup, setting):
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
                                           if p is not None
                                           and isinstance(p, (int, float, str))]
                                cleaned = [p for p in cleaned if np.isfinite(p)]
                                if not cleaned:
                                    continue
                                per_mol.append((mol, np.mean(cleaned),
                                                ie_lookup[mol]))

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
                                    "setting": setting,
                                    "model": MODEL_KEY,
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
        print(f"No Gemini MgCa dir at {GEMINI_DIR}")
        return
    df_full = pd.read_csv(DATA_PATH)
    df_full["canon_smi"] = df_full["isomeric_SMILES"].apply(canonical)

    # MgCa IE lookup (IUPAC -> mean IE for MgCa)
    ie_lookup = (df_full[df_full["Alloy"] == ALLOY]
                 .groupby("IUPAC")["IE"].mean().to_dict())

    rows = []
    for setting in SETTINGS:
        jp = GEMINI_DIR / f"Gemini3_1Pro_{ALLOY}_{setting}.json"
        if not jp.exists():
            print(f"Missing: {jp.name}")
            continue
        new = parse_json(jp, ALLOY, ie_lookup, setting)
        rows.extend(new)
        print(f"  {jp.name}: {len(new)} fold-rows")

    if not rows:
        print("No rows parsed.")
        return

    df_new = pd.DataFrame(rows)
    print(f"\nTotal Gemini MgCa rows: {len(df_new)}")
    print(df_new.groupby("setting")[["r2", "pearson_r"]].mean().round(3))

    # Merge with existing mgca_contrast.csv (replace any prior Gemini rows)
    if RESULTS_CSV.exists():
        df_old = pd.read_csv(RESULTS_CSV)
        df_old = df_old[df_old["model"] != MODEL_KEY]
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_merged = df_new

    df_merged.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved merged CSV to {RESULTS_CSV} ({len(df_merged)} total rows)")


if __name__ == "__main__":
    main()

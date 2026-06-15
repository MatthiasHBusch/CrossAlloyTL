import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"results"/"experiment_results.csv").dropna(subset=["pearson_r"])
FITTED=["RF","GBR","MLP","kNN_Tan","ChemProp"]; MODELS=FITTED+["Gemini3.1Pro"]
SET=["exact","close_filt","close_unfilt","far_filt","far_unfilt"]
def cell(m,s,metric):
    sub=df[(df.model==m)&(df.setting==s)][metric]
    return sub.mean(), sub.std()/np.sqrt(len(sub))
for metric in ["r2","pearson_r","mae"]:
    print("==="+metric+"===")
    for m in MODELS:
        cells=[]
        for s in SET:
            mu,se=cell(m,s,metric)
            cells.append(f"{mu:+.2f}±{se:.2f}" if metric!="mae" else f"{mu:.0f}±{se:.0f}")
        print(f"{m:14s} "+" | ".join(cells))
    fm=[]
    for s in SET:
        mus=[cell(m,s,metric)[0] for m in FITTED]
        fm.append(f"{np.mean(mus):+.2f}" if metric!="mae" else f"{np.mean(mus):.0f}")
    print(f"{'mean(fitted)':14s} "+" | ".join(fm))

# MgCa
mg = pd.read_csv(ROOT/"results"/"mgca_contrast.csv").dropna(subset=["pearson_r"])
def mcell(m,s,metric):
    sub=mg[(mg.model==m)&(mg.setting==s)][metric]
    return sub.mean(), sub.std()/np.sqrt(len(sub))
print("\n=== MgCa pearson_r ===")
for m in MODELS:
    cells=[f"{mcell(m,s,'pearson_r')[0]:+.2f}±{mcell(m,s,'pearson_r')[1]:.2f}" for s in SET]
    print(f"{m:14s} "+" | ".join(cells))
print("=== MgCa Gemini r2/mae ===")
for metric in ["r2","mae"]:
    cells=[f"{mcell('Gemini3.1Pro',s,metric)[0]:+.2f}±{mcell('Gemini3.1Pro',s,metric)[1]:.2f}" for s in SET]
    print(f"{metric:14s} "+" | ".join(cells))
maxr = max(mg[mg.model.isin(FITTED)].groupby(["model","setting"])["pearson_r"].mean())
print("max fitted MgCa r:", round(maxr,2))

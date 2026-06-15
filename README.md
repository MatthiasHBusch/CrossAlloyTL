# CrossAlloyTL — cross-alloy transfer learning for Mg corrosion inhibitors

Code, data, and manuscript sources for:

> **Different Alloy, Different Story? Impact of Transfer Learning on
> Predictive Models for In Silico Screening of Mg Dissolution Modulators.**
> M. Busch, M. Tacke, S. V. Lamaka, M. L. Zheludkevich, C. J. Cyron,
> R. C. Aydin, C. Feiler (2026). *Corrosion Science* (submitted).

## Summary

We probe cross-alloy transfer learning (TL) for predicting the
inhibition efficiency (IE) of small organic dissolution modulators on
magnesium alloys, using the Mg subset of the
[ExCorr](https://excorr.web.app/) electrochemical
corrosion database.

Augmenting a single-alloy training set (60 records) with data from other
Mg alloys raises the mean test-set Pearson correlation from `r ≈ 0.25`
(exact) to `r ≈ 0.83` (close transfer), and the coefficient of
determination from `R² ≈ -0.13` to `R² ≈ +0.39`. We show this gain is
**not** transferable structure–activity (SAR) knowledge: under a
**molecule-disjoint** protocol — every record whose canonical SMILES
coincides with a test molecule is removed from the source pool — the
gain collapses back to the exact-only baseline for every fitted model
(RF, GBR, MLP, kNN-Tanimoto, ChemProp D-MPNN). The apparent TL benefit
is a near-monotone per-alloy IE re-scaling of molecules already measured
on a correlated alloy.

The cause is a tightly correlated SAR landscape: seven Mg materials
(AM50, AZ31, AZ91, E21, WE43, ZE41 and high-purity HPMg51) form a
cluster with pairwise Spearman `r = 0.67–0.92` in inhibitor rankings.
Mg-0.8Ca, the one alloy outside the cluster (`r ≤ 0.35`), is used as a
contrast: there cross-alloy TL not only fails to help but actively
miscalibrates predictions.

An in-context large language model (Gemini 3.1 Pro) is the only method
that retains positive `R²` in the filtered / small-data regime; a
zero-shot control shows that memorisation is probably not the reason.

## Repository layout

```
CrossAlloyTL/
  data/
    ExCorrDatasetClean.csv         Mg + non-Mg ExCorr subset used here (5,167 records)
  src/                             Python + Julia source
    data_loader.py                 Dataset loading, pool construction, molecule-disjoint filter
    models.py                      5 fitted model families (RF, GBR, MLP, kNN-Tan, ChemProp)
    run_experiments.py             Local experiment runner (main, 3 target alloys)
    mgca_contrast.py               Mg-0.8Ca contrast runner
    run_one_combo.py / mgca_run_one.py    Single-combination cluster wrappers
    cluster_submit.py / cluster_submit_mgca.py    SLURM submission
    download_and_aggregate.py / download_mgca.py  Collect cluster results
    run_gemini*.jl                 Gemini in-context / zero-shot runners (see note below)
    julia_lib/                     Vendored Julia deps for the runners (LLM client, prompts; keys via ENV)
    parse_gemini_results.py / parse_gemini_mgca.py   Merge Gemini predictions into the CSVs
    aggregate_pooled_pvalues.py    Pooled Pearson p-values (Table S14)
    cluster_robustness.py          Bootstrap CIs + anchor-exclusion (Table S7, Fig S2)
    make_figures.py                Main Fig 1, Fig 2 (leakage), zero-shot fig, SI figs/tables
    make_mgca_figure.py            Mg-0.8Ca contrast figure
    make_fig1_heatmap.py           Alloy-correlation heatmap (Fig 3)
    am50_vs_cluster.py / alloy_pair_scatters_si.py   SI IE-scatter figures
    sample_counts.py               Per-alloy counts (Table S1)
    paper_numbers.py               Recompute every aggregate quoted in the paper (verification)
    dataset_counts.py              Exact dataset/pool counts (verification)
  results/
    experiment_results.csv         Aggregated main metrics (450 rows: 6 models x 5 settings x 15 folds)
    mgca_contrast.csv              Aggregated Mg-0.8Ca metrics
    sample_counts.csv              Per-alloy record / unique-molecule counts
    test_pool_overlap.csv          Per-fold test/pool molecule overlap
    predictions/                   Per-(alloy,fold,setting,model) molecule-level predictions
    jobs/ , mgca_contrast/         Per-combination metric JSONs
    gemini/ , gemini_mgca/ , gemini_zeroshot/   Raw Gemini outputs + chat transcripts
  figures/                         Final manuscript & SI figures (.png/.pdf)
  SI/                              Generated SI tables (.csv/.tex)
  manuscript/
    manuscript.pdf                 Compiled main paper
    supplement.pdf                 Compiled supplementary information
  requirements.txt , LICENSE , CITATION.cff
```

## Data

`data/ExCorrDatasetClean.csv` is the cleaned subset of the ExCorr
corrosion-inhibitor database used in this study (Mg target/close pools
plus the non-Mg far pool). If you use the dataset, please cite the
ExCorr database paper (Winkler et al., 2025) in addition to this work.

## Reproduction

### Environment

```bash
conda create -n crossalloy python=3.11 -c conda-forge \
    numpy scipy pandas scikit-learn rdkit matplotlib
conda activate crossalloy
pip install -r requirements.txt   # adds chemprop, lightning, torch for the D-MPNN
```

All paths are repository-relative; run scripts from the `src/` directory.

### Reproduce the analysis from the provided results

The committed `results/`, including the raw Gemini outputs, regenerate
every figure, SI table and quoted number without re-running any model:

```bash
cd src
python paper_numbers.py             # prints every aggregate used in the paper
python make_figures.py              # Fig 1, Fig 2, zero-shot fig, SI Fig S5, SI metric tables
python make_mgca_figure.py          # Mg-0.8Ca contrast figure
python make_fig1_heatmap.py         # alloy-correlation heatmap (Fig 3)
python cluster_robustness.py        # cluster-robustness table + forest plot (Table S7, Fig S2)
python am50_vs_cluster.py           # SI Fig S3
python alloy_pair_scatters_si.py    # SI Fig S4
python aggregate_pooled_pvalues.py  # pooled Pearson p-values (Table S14)
```

### Re-run the models from scratch

```bash
cd src
# quick smoke test (one alloy, one setting, no ChemProp)
python run_experiments.py --alloy AZ31 --setting exact --no-chemprop --n-iter 5 \
    --out ../results/smoke.csv

# full local main experiment (slow; ChemProp dominates runtime)
python run_experiments.py --out ../results/experiment_results.csv
python mgca_contrast.py             # Mg-0.8Ca contrast
```

`cluster_submit.py` / `cluster_submit_mgca.py` submit the same jobs to a
SLURM cluster (one job per (alloy, fold, setting, model) combination);
edit the host/user/path constants at the top before use.

### Gemini in-context predictions

The Gemini 3.1 Pro in-context and zero-shot predictions were produced
with the Julia harness (`src/run_gemini*.jl`), which calls Gemini via the
OpenRouter API. The supporting Julia libraries (LLM client, file helpers
and the corrosion prompt templates, reused from our earlier study
`busch2025`) are vendored under `src/julia_lib/`. To re-run the Julia harness set:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."   # only this is needed for the paper
julia src/run_gemini.jl                     # main 3-alloy in-context runs
julia src/run_gemini_mgca.jl                # Mg-0.8Ca contrast runs
julia src/run_gemini_zeroshot.jl            # zero-shot contamination check
```

You do **not** need to re-query the model to reproduce the paper: the
**raw model outputs are included** under `results/gemini*/`, and
`parse_gemini_results.py` / `parse_gemini_mgca.py` reproduce the merged
metrics from them.

### Manuscript

The compiled main paper and supplementary information are provided as
PDFs in `manuscript/` (`manuscript.pdf`, `supplement.pdf`).

## Citation

See `CITATION.cff`, or cite the paper above. Currently submitted.

## License

Code: MIT (see `LICENSE`). For the dataset, please additionally cite and
respect the terms of the ExCorr database.

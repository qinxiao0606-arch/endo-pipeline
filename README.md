# Endometriosis risk pipeline — iGEM ORION / LPCT (CNRS UMR 7019)

Research question: **how do you build and evaluate an endometriosis risk model
when the negative label itself is unreliable?**

## Running

```bash
cd src
python run.py                       # our own pipeline (synthetic data by default)
python run.py --fast                # skip tuning and feature selection
python run.py --input ../data/real/497.csv    # real data — only this changes

python contamination.py             # contamination sweep, synthetic (truth known)
python goldstein.py                 # contamination sweep, Goldstein & Cohen real data
```

## Modules

| File | Role |
|---|---|
| `config.py` | Reads the data dictionary. No variable name is hard-coded anywhere else. |
| `synth.py` | Synthetic generator (Gaussian copula, correlations from Zielinski 2023). |
| `validate.py` | Quality controls C-01 to C-13. |
| `encode.py` | Alias normalisation, three NA types, ordinal vs one-hot, leakage guard. |
| `eda.py` | Case/control comparison, Cramer's V. |
| `model.py` | Baselines, repeated CV, full metrics, tuning, RFECV, sensitivity analyses. |
| `unsupervised.py` | PCA and clustering. |
| `run.py` | Orchestration, 11 stages. |
| `contamination.py` | Contamination sweep on synthetic data — absolute levels. |
| `goldstein.py` | Replication + contamination sweep on real published data — slope. |

## The two contamination experiments

|  | baseline truth | what it measures |
|---|---|---|
| `contamination.py` | known, zero by construction | absolute level |
| `goldstein.py` | unknown | slope per unit added |

Both run a parametric sweep (0 / 5 / 10 / 20 / 30 / 40%) with many random draws
per level; figures show the mean and the 5th–95th percentile band.

## What is NOT measured

`DELTA` in `synth.py` holds effect sizes set from literature orders of
magnitude. These are documented ASSUMPTIONS, not estimates. They exist to make
the synthetic data behave plausibly, not to simulate epidemiological truth.

The correlations taken from Zielinski 2023 cover only 9 of our 22 variables and
contain no correlation with the target. The rest of the matrix is hypothesised
(`EXTRA_CORR`).

## Language

Code, output and figures are in English. The data dictionary
(`data/Dictionnaire_donnees_endometriose_v2.xlsx`) and the questionnaire remain
in French, because they are also the working documents for the French
participants, the partner associations and the laboratory DPO.

## Data

`data/real/` and `data/external/` are gitignored. Real questionnaire responses
must not be published until control C-08 (k-anonymity >= 5) passes and the DPO
has signed off. See the README in each folder.

## Optional dependencies

`shap` and `lightgbm` are not required. Variable importance uses permutation
importance and boosting uses `HistGradientBoostingClassifier`. Install both to
swap them in.

# Bitcoin Transaction Fraud Detection (BT4012 Kaggle Competition)

Rank Bitcoin transactions by how likely they are to be illicit, using the transaction features and the transaction graph.
The leaderboard metric is **ROC-AUC pooled over all test rows**.

## Results

| Submission | Model | Public ROC-AUC |
|---|---|---|
| (project baseline) | NA | about 0.92 |
| `sub1_baseline.csv` | LightGBM on the raw features | 0.94894 |
| **`sub2_stack2.csv`** | LightGBM + neighbour-score stacking ("stack2") | **0.95157** |
| `sub3_blend.csv` | Rank average of sub1 and sub2 | 0.95094 |
| `sub4_stack2_shallow.csv` | Stack2 with a shallow final model (rejected) | 0.94725 |

**Final entries: `sub2_stack2.csv` and `sub3_blend.csv`.**

The final method (stack2) works in two stages. Stage 1 gives every transaction a fraud score from a gradient-boosting model, built out-of-fold inside each training window so that no label leaks.
Stage 2 adds that score and its averages over the transaction's graph neighbours to the original features and trains the final LightGBM.
During training a random share of the neighbour features is blanked out, because 25% of test transactions have no known neighbours.

## Data
Download them from the competition page and put them in a `data/` folder:

| File | Content |
|---|---|
| `train.csv` | time steps 1-35: 165 anonymised features, `time_step`, and a label (1 = illicit, 0 = licit, empty = unknown) |
| `test.csv` | time steps 36-49: same features, no label |
| `txs_edgelist.csv` | directed edges `txId1 -> txId2` between transactions |
| `sample_submission.csv` | the row order the submission file must follow |

## Repository layout

```
bt4012_kaggle_competition/
├── README.md
├── requirements.txt
├── data/                      # competition files (see above)
├── notebooks/
│   ├── 01_setup_and_eda.ipynb
│   ├── 02_feature_engineering_and_modelling_final.ipynb
│   └── 03_final_model.ipynb
├── sub1_baseline.csv          # written by notebook 02, section 12
├── sub2_stack2.csv            # written by notebooks 02 and 03
├── sub3_blend.csv             # written by notebooks 02 and 03
├── sub4_stack2_shallow.csv    # experiment only (appendix of notebook 02)
├── experiment_log.csv
└── holdout_log.csv
```

The notebooks read the data from `../data` and write their output files one level above the data folder (the `project/` folder above), so start Jupyter from inside `notebooks/`.
If the data folder is somewhere else, set `DATA_DIR` in the setup cell of each notebook.

## Setup

Both notebooks were run on Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate    
pip install -r requirements.txt
jupyter notebook
```

`torch` and `torch_geometric` are only needed for the graph neural network experiments. A CPU-only installation is enough.

## Notebooks

Run them in this order (notebook 03 does not depend on 02), each with "Restart & Run All".

### 01 `01_setup_and_eda.ipynb`: setup and exploratory data analysis
Data quality, label and volume trends over time, feature redundancy and class separation, adversarial validation (train vs test drift), and graph analysis (degrees, connected components, phantom nodes, homophily). It ends with a summary of the decisions carried forward.

### 02 `02_feature_engineering_and_modelling_final.ipynb`: feature engineering and modelling
The full experiment record, from baselines to the final model. Sections:
1. Setup and data
2. Validation design
3. Evaluation harness
4. Baselines
5. Drift handling
6. Graph feature engineering
7. Graph diagnostics
8. Neighbour-score stacking (the final method), including a tree-count check
9. Ensembles of tree models
10. Further experiments (all rejected)
11. Frozen holdout, threshold tuning, error analysis
12. Final fit and submission files
13. Decision log

Section 12 writes the submission CSVs. The decision log at the end lists every idea tried, its validation score and whether it was kept.

### 03 `03_final_model.ipynb`: final model only
A short, self-contained notebook that rebuilds the final entries without any experiments or validation: it loads the data, drops duplicate columns, builds the graph, fits plain LightGBM and the stack2 model on all labelled rows (3 seeds), and writes `sub2_stack2.csv` and `sub3_blend.csv`. It needs only `numpy`, `pandas`, `scipy`, `scikit-learn` and `lightgbm`. If submission files from notebook 02 are already on disk, it prints how closely the new files match them.

## Validation

Random k-fold is not used. Folds are split by time step:

| Fold type | (train up to, validate from, validate to) | Purpose |
|---|---|---|
| roll | (14, 15, 20), (20, 21, 26), (26, 27, 30) | validation right after the training window; optimistic |
| gap | (12, 19, 24), (18, 25, 30) | a gap before validation; closest to the test setting, so it is the main metric |
| gapx | (9, 16, 21), (15, 22, 27), (21, 28, 30) | three extra gap folds, used to confirm candidate changes |
| holdout | (25, 31, 35) | frozen; used only for a few finalists |

Fold-to-fold noise is about 0.05 and seed-to-seed noise about 0.005, so a change is treated as real only if it gains at least 0.005 on the gap folds with dev not lower and wins at least 7 of the 8 folds.
Models are ranked by the gap folds first, then by the all-dev score.

## Main findings

- Almost all of the gain over the baseline comes from switching to LightGBM. Neighbour-score stacking adds a small further gain (+0.0026 on the public leaderboard).
- Validation has to resemble the test: next-block folds score about 0.977, folds with a gap about 0.90.
- The graph is strongly homophilic, so the predictions of a transaction's neighbours are informative.
- The test graph is sparser than the training graph (1.66 vs 2.87 neighbours per row; 25.4% of test rows have no known neighbours). Structural features such as PageRank, k-core and component size drift and hurt.
- Many ideas did not help: class weighting, rank normalisation, recency weights, dropping drifting features, tuned LightGBM settings, self-training, unlabelled rows as negatives, sliding windows, bagging, richer neighbour features, a second stacking round, GNNs and blends of tree models.
- Selection noise is real: a shallow final model looked better on validation and then scored lower on the leaderboard.

Note: the "decisions carried forward" at the end of notebook 01 are the plan made before modelling. Some were revised in notebook 02.

## Reproducibility

- All models take an explicit random seed. The submitted files use `QUICK = True`: `N_TREES = 150` (LightGBM and XGBoost use 2 x `N_TREES` = 300 trees), one seed in single-run comparisons, three seeds in the headline comparisons and in the final fit.
- `QUICK = False` uses 400 trees and five final seeds. It is slower, and the scores shift slightly. A tree-count check in notebook 02 (section 8.3) showed that more trees do not improve the score.
- Scores depend on library versions. Install from `requirements.txt` for the best match.
- The full run of notebook 02 takes hours on a laptop CPU; the stacking experiments and the GNN section take the longest.

## Limitations

- The public leaderboard uses part of the test rows, so the private ranking can differ. Differences of a few thousandths between submissions should not be over-interpreted.
- The test period includes events that no validation fold can reproduce.
- The selected model's gain over plain LightGBM is modest (+0.008 on the gap folds, concentrated in one fold) and should be read as such.

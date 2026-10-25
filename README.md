# Bitcoin Transaction Fraud Detection (BT4012 Kaggle Competition)

Rank Bitcoin transactions by how likely they are to be illicit, using the transaction features and the transaction graph.
The leaderboard metric is **ROC-AUC pooled over all test rows**.

## Results

| Submission | Model | Public ROC-AUC |
|---|---|---|
| `sub1_AF_lgbm10.csv` | LightGBM, features A+F (neighbour-score stacking), no sampling, 10 seeds | 0.95193 |
| **`sub2_A_smote4.csv`** | **LightGBM, 158 features (A), SMOTE x4 inside each time step, 10 seeds** | **0.95299** |

**Final entries: `sub1_AF_lgbm10.csv` and `sub2_A_smote4.csv`.**

For reference, public leaderboard score for LightGBM on the raw features: 0.94894

| Evaluation of the final pipeline | ROC-AUC |
|---|---|
| Development folds D1-D5 (mean of five) | 0.9800 (honest level about 0.979) |
| Frozen holdout, steps 32-35, scored once | 0.9985 |
| Public leaderboard (`sub2_A_smote4`) | 0.95299 |

The final pipeline: take the 164 base features after removing one duplicate, drop the six features that move in a straight line with the time step (158 remain), oversample the illicit class with SMOTE **inside each time step** (illicit count x4, 5 neighbours), and fit LightGBM (200 trees, learning rate 0.05, 31 leaves) on all labelled rows of steps 1-35, averaging 10 seeds. It uses no graph feature, no tuning change and no blend.

## Data

Download the files from the competition page and put them in a `data/` folder:

| File | Content |
|---|---|
| `train.csv` | time steps 1-35: 165 anonymised features, `time_step`, and a label (1 = illicit, 0 = licit, empty = unknown) |
| `test.csv` | time steps 36-49: same features, no label |
| `txs_edgelist.csv` | directed edges `txId1 -> txId2` between transactions (always inside one time step) |
| `sample_submission.csv` | the row order the submission file must follow |

## Repository layout

```
bt4012_kaggle_competition/
├── README.md
├── requirements.txt
├── data/                           # competition files (see above)
└── notebooks/
    ├── 01_setup_and_eda.ipynb
    ├── 02_feature_engineering_and_modelling_OLD.ipynb      # older version of the main notebook
    ├── e1156396.ipynb                                      # main notebook, the whole project, sections 1-11
    ├── submissions/
    |    ├── submission_note.txt                             # note on files submitted
    |    ├── sub2_A_smote4.csv
    |    └── sub5_smote_x4.csv
    └── outputs/                                            # written by the notebook
        ├── sub1_AF_lgbm10.csv
        ├── sub2_A_smote4.csv
        ├── submission_log.csv      
        ├── feature_log.csv, sampling_log.csv, tuning_log.csv, ensemble_log.csv
        └── holdout_result.csv
```

The notebook finds the data in `../data`, `./data`, `data` or `../../data`. If it is elsewhere, set `DATA_DIR` in the setup cell. Output files go to `outputs/` next to the notebook, so start Jupyter from inside `notebooks/`. Rename the notebook in this layout if yours has another name.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Required: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `matplotlib`. Optional: `xgboost` (XGB in sections 5 and 9, skipped if missing) and `torch` (the two graph neural networks in section 5 and GraphSAGE in section 9; a CPU-only install is enough, no extra graph library is used). Use a recent Python 3 (the earlier notebooks were run on 3.13).

## The notebook

One notebook, run top to bottom (restart and run all). It is organised by decision, in a fixed order, and every section ends with a summary of results, findings and decisions.

| Section | Content | Outcome |
|---|---|---|
| 1 | Setup and data | |
| 2 | Diagnosis of the graph and features: sparsity, homophily, feature and concept drift | test graph is a sparse, degree-biased sample (25% of rows isolated); node dropout q = 0.42 chosen; six time-index features drift |
| 3 | Duplicate and near-duplicate features | 1 feature dropped (164 remain) |
| 4 | Validation and evaluation design | five rolling folds, frozen holdout, seed noise 0.0020, decision rule |
| 5 | Model family: LR, RF, HGB, XGB, LightGBM, GCN, GraphSAGE | LightGBM best (0.9747) |
| 6 | Feature engineering by groups (A, B, C, D, D2, G, E, E2, F, F2) | six time-index features dropped (A); graph features not kept in the end |
| 7 | Class imbalance: class weights, under- and oversampling, SMOTE | per-step SMOTE x4 helps; the rest lose; A only + SMOTE x4 carried forward (0.9800) |
| 8 | Tuning: 21 tests | none adopted; defaults stay |
| 9 | Ensembling: 7 blends | none adopted |
| 10 | Holdout, scored once | 0.9985 |
| 11 | Final fit and submission files | `sub2_A_smote4.csv`, checked against `sample_submission.csv` |

The holdout can be scored only once per session; a second call is refused. Change a setting and the notebook must be restarted before the holdout is rerun.

## Validation

Random k-fold is not used. Folds are split by time step, with a 4-step gap between training and validation:

| Fold | Train steps | Gap | Validation steps | Illicit rows in validation |
|---|---|---|---|---|
| D1 | 1-11 | 12-15 | 16-19 | 359 |
| D2 | 1-15 | 16-19 | 20-21 | 360 |
| D3 | 1-17 | 18-21 | 22-24 | 348 |
| D4 | 1-20 | 21-24 | 25-28 | 323 |
| D5 | 1-24 | 25-28 | 29-31 | 518 |
| Holdout | 1-27 | 28-31 | 32-35 | 584 |

* The test graph is sparser than the training graph (25.4% of test rows have no observed neighbour), so validation hides each neighbour with probability 0.42 (node dropout). This matches the test mean degree (1.656 vs 1.660); the isolated share is slightly low (21.9% vs 25.4%).
* ROC-AUC on seed-averaged predictions (3 seeds) is the only decision metric; PR-AUC is information only.
* The seed noise level is 0.0020. A change is adopted only if the mean gain exceeds it, it wins at least 4 of 5 folds, and the best fold supplies at most 50% of the gain. Failed ideas stay in their section marked "not adopted".

## Main findings

* LightGBM on the tabular features is the strongest base model (0.9747), ahead of XGB (0.9704), HGB (0.9690), GraphSAGE (0.9669) and the weaker models.
* Nearly all gains came from two simple steps: dropping the six time-index features (0.9747 to 0.9771) and per-step SMOTE x4 (to 0.9800, honest level about 0.979). Class weights, undersampling and plain oversampling all lost; SMOTE across steps lost 0.0032 against SMOTE inside each step.
* The graph is strongly homophilic, but graph structure features, neighbour aggregates and neighbour-score stacking added little on top of the base features (the base features probably already contain neighbour aggregates, which was not tested). The stacking pair A+F passed the rule, but F stopped helping once SMOTE was used, so the final pipeline does not use it.
* 21 tuning tests and 7 blends produced no change that passed the rule.
* Much of the gain from A and SMOTE comes from one hard fold (D3, steps 22-24).

## Reproducibility

* All models take an explicit random seed. Validation uses seeds (0, 1, 2); the final fit and the holdout use 10 seeds. Defaults (200 trees, learning rate 0.05) are fixed once in section 1.
* Scores depend on library versions; install from `requirements.txt` for the best match.
* Features and graph blocks are cached within a session. The slowest parts are the GNN runs in section 5 (457 s and 1,209 s per model run) and the stacking tests in section 6.

## Limitations

* Validation (about 0.98) is well above the public score (about 0.953). The test period is sparser, drifts, and the public board covers only a subset of the test rows, so differences of a few thousandths between submissions should not be over-interpreted.
* The folds and seeds were reused for well over 50 decisions, so development numbers are somewhat optimistic. The honest level of the final pipeline is about 0.979 (fresh-seed checks).
* The very high holdout score (0.9985) covers only four time steps and should not be read as the expected test score.
* XGB, HGB and GraphSAGE were not tuned, so a better ensemble was not ruled out.

# Poker Action Modeling

A machine-learning project for learning a player's poker decision patterns from hand histories. The current prototype focuses on **preflop action classification** and compares interpretable classical models before introducing more complex methods.

> **Project status:** Research prototype. The model-training workflow is functional, but the hand-history parser and target construction require additional validation before the reported performance should be treated as final.

## Project objective

The immediate objective is to predict the player's action from information available at decision time. The current dataset represents four actions:

- `fold`
- `check`
- `call`
- `raise`

The broader goal is to produce a model that can imitate the player's decision distribution—not merely select the most frequent action. Future versions may use a strategic three-class representation:

- `fold`
- `passive` (`check` or `call`, depending on the legal state)
- `aggressive` (`bet` or `raise`, depending on the legal state)

Bet or raise sizing would then be handled by a separate conditional model.

## Why start with preflop?

Preflop modeling provides a manageable first end-to-end problem:

- Every decision has two private cards and a table position.
- No board-card representation is required.
- The feature space is smaller than on later streets.
- It allows the full pipeline—parsing, validation, preprocessing, training, tuning, and testing—to be established before increasing complexity.

Flop, turn, and river decisions are intentionally outside the current modeling scope.

## Current modeling workflow

The current experiment follows this sequence:

1. Parse hand histories into decision-level rows.
2. Select preflop observations.
3. Construct action labels.
4. Create development and held-out test sets.
5. Apply preprocessing inside a scikit-learn `Pipeline`.
6. Tune models using repeated stratified cross-validation.
7. Select models using cross-validated macro F1.
8. Inspect balanced accuracy, log loss, class-level performance, and train-validation gaps.
9. Evaluate the selected model on held-out data.

Keeping preprocessing inside the model pipeline ensures that encoders and scalers are fitted only on the relevant training fold during cross-validation.

## Models

Three models currently define the comparison:

### Dummy classifier

A majority-class baseline. A useful model must substantially outperform it.

### Multinomial logistic regression

The initial interpretable baseline. Hyperparameters currently include:

- Inverse regularization strength (`C`)
- Optional balanced class weights

Logistic regression provides a stable reference point but may underfit nonlinear poker relationships and feature interactions.

### Decision tree

The first nonlinear comparison. Hyperparameters currently include:

- Maximum tree depth
- Minimum observations per leaf
- Optional balanced class weights

The tree can learn conditional rules and interactions automatically, but it has a greater risk of overfitting the relatively small dataset.

## Evaluation strategy

### Primary metric: macro F1

The action classes are imbalanced, with folds representing approximately half of the current preflop observations. Accuracy alone would allow a model to appear successful by predicting `fold` too frequently.

Macro F1 calculates F1 separately for each action and gives every class equal influence:

```text
macro F1 = mean(F1_call, F1_check, F1_fold, F1_raise)
```

This prevents the dominant fold class from hiding failures on less common actions. Its limitation is that the rare `check` class currently receives disproportionate influence and produces high fold-to-fold variance. The planned strategic action representation would address this more naturally than deleting check observations.

### Secondary metrics

- **Balanced accuracy:** average recall across actions
- **Log loss:** quality of the full predicted probability distribution
- **Per-class precision and recall:** identifies which actions improved or deteriorated
- **Normalized confusion matrix:** shows the specific action confusions
- **Train-validation gap:** helps identify overfitting
- **Accuracy:** retained as a familiar secondary summary, not the selection criterion

All candidate models must use the same cross-validation splits and preprocessing rules.

## Provisional results

The latest experiment produced the following repeated cross-validation results:

| Model | CV macro F1 | CV standard deviation | Train macro F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Dummy classifier | 0.168 | 0.001 | 0.168 | 0.250 |
| Logistic regression | 0.549 | 0.085 | 0.607 | 0.540 |
| Decision tree | 0.569 | 0.081 | 0.755 | 0.620 |

The decision tree is the provisional leader on macro F1 and balanced accuracy, suggesting that nonlinearities or interactions may be useful. However:

- Its macro-F1 advantage over logistic regression is only about 0.02.
- Its train-validation gap is substantially larger, indicating more overfitting.
- The standard deviation across validation folds is large relative to the difference between the models.
- The rare check class makes model rankings unstable.

The current evidence therefore supports treating logistic regression and the decision tree as competitive rather than declaring a final winner. Probability quality and class-level errors must also be considered.

## Data and features

The current preflop dataset contains approximately 707 decisions. Candidate inputs include:

- Hero's two hole-card ranks
- Position
- Facing-bet ratio
- Suit relationship
- Rank-connectivity features

The following fields are target-derived, post-outcome, constant, or otherwise unsuitable as preflop predictors and should be excluded:

- `if_fold`
- `hero_bet_ratio`
- `result`
- Board-card columns
- Later-street indicators
- Constant preflop fields such as `stage_pre` and the current `gut_po`

Future feature engineering should canonicalize the two hole cards into features such as:

- High rank
- Low rank
- Pair indicator
- Suited indicator
- Rank gap

This avoids treating equivalent hands differently because of card order.

## Known data limitations

The current scores are provisional because several parser and labeling issues remain.

### Action reconstruction

The parser identifies the original action text but currently discards it. The modeling notebook reconstructs actions from fold and bet-ratio fields. The parser should instead preserve the original normalized action directly.

### Blind commitments and facing ratio

The current parser initializes Hero's street commitment to zero and skips blind-posting lines. As a result, a player in the blinds may appear to face a positive amount when checking is available. The uploaded dataset currently contains check-labeled observations with positive `facing_ratio` values.

The parser should track each player's existing street commitment and calculate:

```text
amount to call = current maximum street commitment
                 - Hero's existing street commitment
```

### Stable hand identity

The current split is based on a row index. A future dataset should preserve:

- Hand identifier
- Hand timestamp
- Session identifier when available

This will allow grouped or chronological splitting and prevent decisions from the same hand from crossing between training and evaluation sets.

### Limited rare-class data

There are currently very few check observations. This makes per-class metrics and macro F1 unstable. More data—or a strategic action representation that merges check with call—is required before making strong conclusions about that behavior.

## Repository structure

```text
Poker_Neural_Network-master/
├── analysis_v1/              # Legacy neural-network and parsing experiments
├── data/
│   └── parsed_save2.csv      # Current parsed dataset
├── src/
│   ├── labels.py             # Current label and mapping helpers
│   ├── preprocessing.py      # Shared ColumnTransformer
│   ├── splits.py             # Dataset splitting helpers
│   └── models.py             # Candidate estimators and search grids
├── data_exploration.ipynb    # Current exploration and modeling notebook
└── README.md
```

The intended structure as the project matures is:

```text
src/
├── parser.py          # Hand histories to validated decision rows
├── labels.py          # Target normalization
├── features.py        # Poker-specific feature engineering
├── splits.py          # Grouped or chronological splitting
├── preprocessing.py   # Leakage-safe transformations
├── models.py          # Estimator definitions and parameter grids
├── evaluation.py      # Shared cross-validation and reports
├── train.py           # Reproducible final training workflow
└── predict.py         # Inference with a saved complete pipeline

notebooks/
├── 01_data_exploration.ipynb
└── 02_model_experiments.ipynb
```

The notebook is the experimental playground. Reusable transformations, model definitions, and evaluation procedures belong in `src/`. Once an experiment is settled, `train.py` should reproduce it without depending on notebook execution order.

## Getting started

Create and activate a virtual environment, then install the current core dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter lab
```

Open `data_exploration.ipynb` to inspect the current data and experiment. A dedicated `02_model_experiments.ipynb` is planned so data-quality analysis and model comparison remain separate.

## Development principles

- Use only information available when the decision was made.
- Preserve the original action label during parsing.
- Split by hand or time before fitting transformations.
- Fit preprocessing only through the complete model pipeline.
- Compare all models using identical cross-validation splits.
- Select models using training-set cross-validation, not the final test set.
- Report uncertainty and class-level behavior, not only a single average score.
- Prefer a simpler model when performance differences are small relative to validation variability.
- Save the complete fitted pipeline, including preprocessing.

## Roadmap

### Data integrity

- [ ] Preserve exact action names during parsing
- [ ] Correct blind and per-player commitment tracking
- [ ] Validate facing-bet and pot-ratio calculations against sample hands
- [ ] Add stable hand IDs, timestamps, and session IDs
- [ ] Add automated tests for label and feature construction

### Preflop model

- [x] Establish a dummy baseline
- [x] Train and tune multinomial logistic regression
- [x] Train and tune a decision-tree classifier
- [ ] Add log loss to the model-comparison table
- [ ] Compare models fold by fold
- [ ] Canonicalize hole-card features
- [ ] Evaluate the selected model on newly collected hands

### Future modeling

- [ ] Decide between four recorded actions and three strategic action classes
- [ ] Evaluate probability calibration
- [ ] Add a conditional bet-sizing model
- [ ] Expand to flop decisions after the preflop pipeline is validated
- [ ] Consider ensemble models only after establishing clean data and stable baselines

## Scope and disclaimer

This repository is an educational machine-learning project. It is intended to study data preparation, classification, validation, and behavioral imitation. The current models and results should not be interpreted as a proven profitable poker strategy.

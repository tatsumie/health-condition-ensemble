# Health Condition Classification Ensemble

A reproducible multiclass machine-learning pipeline for predicting health-condition labels from mixed numerical and categorical features. The project compares CatBoost and LightGBM, generates honest out-of-fold predictions, and optimizes a probability-weighted ensemble for balanced accuracy.

## Result

The best five-fold out-of-fold blend used **29% CatBoost and 71% LightGBM** and achieved:

| Metric | Score |
|---|---:|
| Balanced accuracy | 0.949460 |
| Accuracy | 0.939587 |
| Macro F1 | 0.866656 |

Balanced accuracy is the primary metric because the 690,088-row training set is highly imbalanced across three target classes.

## What the pipeline does

1. Loads the Kaggle training and test files.
2. Creates separate model views so each estimator receives categorical data in its preferred representation.
3. Trains fresh CatBoost and LightGBM models across five stratified folds.
4. Stores out-of-fold class probabilities for every training row.
5. Evaluates 101 ensemble weights from 0.00 to 1.00.
6. Retrains both models on all labeled data and creates a competition submission.
7. Records the best experiment in SQLite for local tracking.

## Repository structure

```text
.
├── src/
│   └── train_ensemble.py
├── results/
│   └── oof_weight_results.csv
├── .gitignore
├── README.md
└── requirements.txt
```

Raw competition data, submissions, model logs, and local databases are intentionally excluded from version control.

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the competition `train.csv` and `test.csv` files from Kaggle and place them in the repository root. Then run:

```bash
python src/train_ensemble.py
```

The script writes:

- `submission_oof_ensemble.csv`
- `oof_weight_results.csv`
- `accuracyData.db`

## Technical decisions

- **Out-of-fold validation:** Each training row is predicted only by models that did not train on that row.
- **Model-specific categorical handling:** CatBoost receives string categories, while LightGBM receives aligned pandas categorical dtypes.
- **Class imbalance:** Both estimators use balanced class weighting, and ensemble selection uses balanced accuracy.
- **Probability blending:** Class probabilities are combined before selecting the highest-probability class.
- **Reproducibility:** Fold creation and both estimators use fixed random seeds.

## Technologies

Python, pandas, NumPy, scikit-learn, CatBoost, LightGBM, and SQLite.


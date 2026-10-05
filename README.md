# Titanic Survival Classification — From Scratch vs scikit-learn

Predict whether a passenger survived the Titanic from class, sex, age, family size, fare and port of embarkation.
Core algorithms are implemented **from scratch with NumPy only**, then compared against scikit-learn on an identical train/test split.

## What's inside

| Notebook | Content |
|---|---|
| `A_minimal_prep_and_split.ipynb` | Minimal, leakage-safe preprocessing and a stratified 80/20 split |
| `B_from_scratch_models.ipynb` | NumPy-only Logistic Regression (none / L1 / L2), Bagging, AdaBoost (stumps), metrics, CV, ROC/PR curves |
| `C_library_models.ipynb` | scikit-learn Logistic Regression, Random Forest, AdaBoost, Gradient Boosting, HistGradientBoosting + comparison with B |
| `Technical_Report.pdf` | ~1000-word write-up of results and analysis |

## Highlights

- **Correctness check:** analytic gradient matches a numerical gradient (~1e-10); from-scratch logistic regression coefficients match scikit-learn to ~1e-7.
- **L1 vs L2:** L1 (proximal gradient / soft-thresholding) produces exact zero coefficients; L2 only shrinks them.
- **Honest evaluation:** the test set has only 179 rows, so most differences between models are within bootstrap noise.
- **CV vs test:** boosting models led in cross-validation but trailed on the test set — a reminder not to over-trust a single split.

| Model (test set, threshold 0.5) | Accuracy | ROC-AUC |
|---|---|---|
| Bagging of trees (scratch) | 0.816 | 0.846 |
| Logistic Regression L1 (scratch) | 0.816 | 0.844 |
| Logistic Regression L1 (sklearn) | 0.816 | 0.844 |
| Random Forest (sklearn) | 0.816 | 0.838 |
| AdaBoost, stumps (scratch) | 0.799 | 0.795 |
| Gradient Boosting (sklearn) | 0.782 | 0.799 |

![Test ROC-AUC of all models](figs/comparison_auc.png)
![Regularisation paths](figs/B_regularisation_path.png)

## How to run

```bash
pip install -r requirements.txt
jupyter lab        # run the notebooks in order: A -> B -> C
```

Notebook A downloads the Titanic data automatically (a public mirror of the
[Kaggle Titanic dataset](https://www.kaggle.com/c/titanic)) and writes the processed split to `data/`, which B and C reuse.
Seed 42 is used everywhere. Total runtime is about 2.5 minutes on a CPU.

Results were produced with Python 3.12, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, matplotlib 3.10.8.

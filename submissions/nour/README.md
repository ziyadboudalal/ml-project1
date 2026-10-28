# EPFL Machine Learning — Project 1: MICHD prediction


Final model: a blend of our own gradient-boosted decision trees and a Newton-method logistic
regression. 
Cross-validated F1: 0.444
AIcrowd F1: 0.455

## Setup

Install the required packages:

    python -m pip install -r requirements.txt

## Dataset

Download the dataset from:
https://www.aicrowd.com/challenges/epfl-machine-learning-project-1

Extract the CSV files into a folder named `dataset` at the root of the project.

The dataset is not included in this repository. Each team member must download it locally.

## Reproducing the submission

    python run.py

This writes `submission.csv`, identical to our best AIcrowd submission.
All random steps use a fixed seed (42).

## Files

| File | Content |
| `implementations.py` | The 6 required methods: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression` |
| `run.py` | Final pipeline: preprocessing, Newton logistic regression, gradient boosting, blend, submission |
| `project1_notebook.ipynb` | Full analysis: exploratory data analysis, preprocessing, baselines of the 6 methods, cross-validation, improvements, ablation studies, choice of the final settings |
| `helpers.py` | Data loading and submission writing (provided by the course) |
| `documentation/` | Project description and our results log |

## Pipeline

1. **Special codes** (BRFSS 2015 codebook): "don't know / refused" codes (7/9, 77/99, ...) become
   missing values, "none / never" codes (88, 888, 555) become 0, coded frequencies (alcohol, fruit,
   vegetables, exercise) and weight / height are converted to real units.
2. **Logistic regression features**: metadata and columns with more than 80% missing values are
   dropped; columns with at most 30 distinct values are one-hot encoded (rare / missing / unseen
   values share one column, the most frequent category is the reference); continuous columns are
   median-imputed, clipped to the 1st–99th percentiles and standardized; missing-value and
   "answered none" indicators; squares of the 10 and pairwise products of the 25 features most
   correlated with the label. Everything is learned on the training data only.
3. **Logistic regression** (L2, λ = 1e-4) trained with **Newton's method** and a line search.
4. **Gradient-boosted trees** (implemented in NumPy): features binned into 63 quantile bins plus
   one bin for missing values, histogram-based depth-4 trees, learning rate 0.1, half of the rows
   and features per tree, 209 trees (number chosen by early stopping in cross-validation).
5. **Blend**: `p = 0.7 * p_gbdt + 0.3 * p_logistic`, positive if `p >= 0.225`.

## Validation

All settings are chosen by 3-fold cross-validation, with the preprocessing fitted inside each
fold. The decision threshold and the blend weight are chosen on the out-of-fold predictions, since
the leaderboard metric is the F1 score and only 8.8% of the respondents are positive.

| Model | CV F1 | AIcrowd F1 |
| --- | --- | --- |
| Newton logistic regression | 0.4302 ± 0.0028 | 0.440 |
| Gradient-boosted trees | 0.4421 ± 0.0035 | – |
| **Blend (final)** | **0.4436 ± 0.0036** | **0.455** |
 
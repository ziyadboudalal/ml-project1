# CouscousAI - EPFL CS-433 Project 1

**Nour Chebil, Ismail Moudden and Ziyad Boudalal.**

| Start here | Contents |
|---|---|
| [Two-page report](documentation/report_merged/report.pdf) | Methods, baselines, common-fold comparisons, ablation study and AIcrowd result |
| [Analysis notebook](experiments/project1_notebook.ipynb) | The whole project with its outputs: exploration, preprocessing, the six methods, cross-validation, improvements, ablations |
| [Requirements and hand-in](documentation/SUBMISSION.md) | Where each assignment requirement is addressed |
| [Experiment results](documentation/Results.md) | Detailed comparisons and their interpretation |

The delivered system is a NumPy-only blend of histogram gradient-boosted trees
and Newton logistic regression. Its single configuration is `CONFIG` in
`final_pipeline.py`. The final report, figures and common-fold validation use
that configuration: **30-category cutoff, 3 shuffled folds, seed 42**.

## Install and data

```bash
git clone https://github.com/ziyadboudalal/ml-project1.git
cd ml-project1
```

The validated environment is **Python 3.11.9, NumPy 1.26.2 and Matplotlib 3.8.2**.
The code targets Python 3.9+, but the recorded execution used Python 3.11.
Install the pinned dependencies, preferably in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

Download the data from the
[course competition](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
and keep their original filenames:

```text
dataset/                  # local only, excluded from Git
    x_train.csv
    y_train.csv
    x_test.csv
```

The implementation uses only the Python standard library and NumPy; Matplotlib
is used only for figures. No external ML libraries, implementations or datasets
are used by the prediction pipeline.

## Generate the delivered predictions

From the repository root:

```bash
python run.py
```

This loads `artifacts/final_model.npz` and writes `submission.csv` and
`submission.manifest.json`. Only `dataset/x_test.csv` is needed for inference.
Paths are resolved relative to the repository, so the script also works from
another working directory. Alternative input/output locations:

```bash
python run.py --data-dir /path/to/data --output /path/to/submission.csv
```

The loader checks column order and unique IDs; the writer verifies that all test
IDs appear in order and that labels are exactly -1/+1. The manifest records
SHA-256 fingerprints, row counts, configuration, training provenance and the
number of predicted positives. Model archives contain typed NumPy arrays and
JSON metadata, and are loaded with `allow_pickle=False`.

**Confirmed AIcrowd result:** our delivered CSV was successfully graded as
submission **338118** on **28 October 2026**, with **F1 = 0.455** and
**accuracy = 0.886**. We recorded this result from the team's upload confirmation
and AIcrowd result screenshot. The manifest associates it with the exact CSV
SHA-256; another CSV does not inherit this score. These leaderboard metrics
are distinct from our common-fold development F1 of **0.443602**.

To compare every ID and prediction against a copy of the uploaded file:

```bash
python run.py --output artifacts/reproduction/submission.csv --reference submission.csv
```

This writes a separate output and records the number of different predictions
in its manifest (expected: **0**). The delivered CSV has **109,379 rows** and
**13,390 positives**.
Its SHA-256 is recorded in [the manifest](submission.manifest.json).

## Refit the saved model

```bash
python run.py --train
```

This fits all labeled rows, saves the model and reloads it before predicting.
It is substantially slower and uses several GB of memory; ordinary inference
does not retrain. See the manifest for measured training time and environment.
Avoid running training and cross-validation simultaneously on an 8 GB machine.
For consistent CPU usage, `OPENBLAS_NUM_THREADS=2` was used for the final run.

The delivered settings are:

| Component | Setting |
|---|---|
| Logistic encoder | At most 30 distinct values; `_STATE` always categorical |
| Filtering | Over 80% missing or constant; eight bookkeeping columns removed |
| Continuous variables | Training median, 1st/99th percentile clipping, scaling |
| Added features | Missing/none flags, squares of top 10 continuous features, products of top 25 encoded features |
| Logistic training | Newton, max 15 steps, L2 = 1e-4 **including intercept**, line search |
| Trees | 209 trees, depth 4, learning rate 0.1, leaf L2 = 1, minimum child Hessian 5 |
| Binning and sampling | At most 63 value bins plus missing bin; 50% rows/features per tree |
| Prediction | 0.7 times tree probability + 0.3 times logistic probability; threshold 0.225 |

The code retains the notebook's full floating-point blend weight
`0.7000000000000001`. All stochastic operations use the recorded seed 42.

## Full analysis notebook

[`experiments/project1_notebook.ipynb`](experiments/project1_notebook.ipynb) contains the whole
project with its outputs: data exploration, special codes and preprocessing, the six methods from
class (Table I of the report), the cross-validation of the regularization and of the class weights
(Fig. 1), the improvements, both ablation studies (Table III), the gradient boosting and the blend.
It uses the same three folds (seed 42) as `experiments/validate_final.py`, and both give identical
results. A full run takes several hours.

## Reproduce validation and figures

```bash
python experiments/validate_final.py
python documentation/final_figures.py
```

Validation uses the original **three shuffled, non-stratified folds**, seed 42.
All compared methods use identical rows. Each fold fits its own preprocessing,
supervised feature selection and weights. Tree early stopping uses 10% of the
outer training rows, patience 50, maximum 1000 trees. The original final-refit
count was `round(mean([183,171,216]) * 1.1) = 209`.

Out-of-fold (OOF) scores select a threshold on `[0.02,0.98]` with step 0.005.
The final blend is additionally evaluated at its locked threshold 0.225.
Reported SD uses `ddof=0`. Because model choices, thresholds and blend weights
were selected on development OOF scores, these are **development estimates**,
not an independent final test estimate. Fold SD is not a confidence interval.

Completed folds are cached locally. Cached results are reused only when the
dataset and source hashes match; choose another `--output-dir` after changing
the experiment. The scripts produce:

- `artifacts/validation/results.json`: full-precision scores and per-fold metrics.
- `artifacts/validation/protocol.json`: configuration and data/source hashes.
- `artifacts/validation/oof_predictions.npz`: saved OOF predictions, IDs and labels for figures (no raw survey features).
- `documentation/final_figures/`: figures with a provenance record.
- `documentation/report/results_*.tex`: generated numbers and comparison table (used by the first version of the report, in `documentation/report/`).

Saved OOF predictions are included, so `python documentation/final_figures.py`
can regenerate the figures immediately without rerunning training. Source and
configuration fingerprints are checked before plotting. Git attributes preserve
the bytes of fingerprinted files across Windows and Unix checkouts.

Our earlier five-fold and 70/15/15 experiments use different protocols and are
outside the final comparison. We keep the final report's figures tied to the
single common-fold experiment above.

## Report

The [team report](documentation/report_merged/report.pdf) is two pages plus a page of
references, with its [LaTeX source](documentation/report_merged/report.tex) adjacent. Its tables
and two of its figures are regenerated from the recorded results by
`documentation/report_merged/make_report_assets.py`. To build it:

```bash
python documentation/report_merged/make_report_assets.py --out documentation/report_merged
cd documentation/report_merged
pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex
```

Tectonic is an alternative LaTeX compiler. LaTeX tooling is only for building
the report and is not a runtime ML dependency.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests cover stable Newton optimization, nonlinear boosting, deterministic
predictions, special response codes, CSV integrity, exact model round-trips
and attribution of the AIcrowd score only to the confirmed CSV.
The six required methods are in `implementations.py`; least squares forms
double-precision normal equations and handles rank-deficient designs with a
NumPy pseudoinverse of the normal matrix (never `numpy.linalg.lstsq`).

The [course public tests](https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests)
are kept outside this repository. Instructions for running them against the
exact published commit are in [the hand-in guide](documentation/SUBMISSION.md#verification).

## Experiment history and hand-in

[Our results](documentation/Results.md) record the verified common-fold comparison and its
limitations. We present the work collectively under the three team authors.
Earlier individual experiments remain available in the Git history; their useful code was
consolidated into the final pipeline and the analysis notebook.

The course submission consists of the report PDF and a GitHub link to the exact commit
(see [the final hand-in steps](documentation/SUBMISSION.md#final-hand-in)).

# CouscousAI - assignment requirements and hand-in

**Team:** Nour Chebil, Ismail Moudden and Ziyad Boudalal.
**Delivery branch:** `couscousAI-final`.

This guide maps the supplied Fall 2026 Project 1 description (Steps 2-4 and
the report guidelines) to our final repository. The project description is
the source of the requirements; this guide describes our implementation.

## Requirements and evidence

| Assignment requirement | Where it is addressed |
|---|---|
| Public, executable and documented Python code | Public team repository; [README](../README.md) gives installation and execution commands |
| `implementations.py` at the root with the six required methods | [implementations.py](../implementations.py), with the signatures listed below |
| Standard library and NumPy only for ML; visualization libraries only for plots | Prediction and training import only standard-library modules and NumPy; Matplotlib is used in [final_figures.py](final_figures.py) |
| No external datasets or external ML implementations | Training uses only the supplied `x_train.csv` and `y_train.csv`; algorithms are implemented in our Python sources |
| `run.py` reproduces the competition predictions exactly | [run.py](../run.py) loads [saved weights](../artifacts/final_model.npz); the CSV hash and reproduction checks are recorded in [the manifest](../submission.manifest.json) and [delivery verification](../artifacts/delivery_verification.json) |
| Training code supplied if inference uses saved weights | `python run.py --train`; training orchestration is in `run.py` and algorithms in [final_pipeline.py](../final_pipeline.py) |
| Correct test IDs, CSV format and supplied CSV-writing helper | [project_io.py](../project_io.py) validates IDs/order and labels; `run.py` uses `create_csv_submission` from [helpers.py](../helpers.py) |
| Document preparation, feature generation and local validation | [README](../README.md), [results](Results.md), [validation script](../experiments/validate_final.py) and [protocol](../artifacts/validation/protocol.json) |
| Local evaluation, baselines and evidence for improvements | Ridge, Newton, Newton with interactions, boosted trees and blend compared on the same three folds; the report explains the development-selection limitation |
| At most two report pages, no appendix; LaTeX source | [Report PDF](report/report.pdf): two pages including references; [LaTeX source](report/report.tex), class, generated tables and figures are included |
| Report the performance of the delivered submission | AIcrowd submission **338118**, 28 October 2026: **F1 0.455**, **accuracy 0.886**, included in the report |
| Final submission contains the report and a GitHub code link | Follow the final hand-in steps below; use a link to the exact reviewed commit |

## Required method interface

The following functions are importable directly from `implementations.py`:

```python
mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)
mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)
least_squares(y, tx)
ridge_regression(y, tx, lambda_)
logistic_regression(y, tx, initial_w, max_iters, gamma)
reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
```

Each returns `(w, loss)` at the final weights, with one-dimensional weight and
target vectors. MSE includes the factor `0.5`; SGD samples one observation per
update. Logistic regression expects targets in `{0, 1}`. The returned loss
excludes the penalty for both regularized methods. Least squares uses normal
equations and does not call `numpy.linalg.lstsq`.

These required gradient-descent functions are distinct from the Newton solver
used by our final model. Both are supplied. The final model combines tree and
logistic probabilities with weights 0.7 and 0.3, then applies threshold 0.225.

## Verification

From the repository root, with the course test data in `dataset/`:

```bash
python -m unittest discover -s tests -v
python run.py --output artifacts/reproduction/submission.csv --reference submission.csv
python documentation/final_figures.py
```

The first command runs nine pipeline tests. The second checks every predicted
label and ID against the delivered CSV while keeping that reference intact.
The expected output contains **109,379 rows**, **13,390 positives** and SHA-256:

```text
4986920c9485b722c1b2992667dba8aa43f9f0d2122df8572229bcc412e775ce
```

The last command regenerates the report's figure and table from the included
out-of-fold predictions, without retraining. To recompute validation from the
raw training data, run `python experiments/validate_final.py`; this takes much
longer and fits preprocessing separately in each fold.

The [course public tests](https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests)
are kept outside this submission, as their instructions recommend. In their own
grading environment and directory, run the following with `FULL_COMMIT_SHA` replaced by the reviewed
commit's 40-character hash:

```bash
python -m pytest --github_link https://github.com/ziyadboudalal/ml-project1/tree/FULL_COMMIT_SHA .
```

The grading tools and their dependencies belong to that separate environment;
they are not imported by our prediction or training code. Public tests check
part of the requirements; they do not replace the team's code/report review.

## Final hand-in

The supplied statement gives **Thursday 29 October 2026 at 16:00** as the
deadline and [mlcourse.epfl.ch](http://mlcourse.epfl.ch) as the submission site.
AIcrowd evaluation and the final report/code hand-in are separate steps.

1. Review the final branch together, including the report, authors, results and
   reproduction commands. Commit and push any agreed corrections to
   `couscousAI-final`.
2. In that branch, run `git status` to ensure all intended changes are committed
   and pushed. Run `git rev-parse HEAD` to obtain the exact commit hash.
3. Form the permanent code URL
   `https://github.com/ziyadboudalal/ml-project1/tree/FULL_COMMIT_SHA` using that
   hash. Verify that this URL shows the reviewed files. On GitHub, pressing `y`
   on the branch page also produces a commit permalink.
4. Submit [report/report.pdf](report/report.pdf) from that same commit and its
   permanent code URL on the course submission site. Save the submission
   confirmation.

The file list is [DELIVERY_FILES.txt](../DELIVERY_FILES.txt), explained in
[REPO_CONTENTS.md](../REPO_CONTENTS.md). The local ZIP contains the same files
for convenience; the requested code deliverable is the public GitHub link.

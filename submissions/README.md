# Ismail's submission

NumPy only.

```bash
python submissions/ismail.py              # CV, refit, write model.json + submission.csv
python run.py                             # submission.csv from the saved model.json
python -m unittest discover -s tests      # tests
```

## Model

L2-regularized logistic regression (lambda = 3e-4, bias not penalized),
fitted with Newton's method.

Features, with all statistics computed on training rows only:

- Identifiers, dates, state and survey weights are dropped, as are constant
  columns and columns with less than 1% observed values.
- Columns with at most 60 distinct values are one-hot encoded. Answer codes
  such as "don't know" or "refused" keep their own level.
- Other columns are clipped to their 1st/99th percentiles, standardized and
  expanded with a squared term.
- Missing-value indicators.
- Pairwise products of the 20 encoded columns most correlated with the label.

Positives are rare (8.8%), so the decision threshold maximizes F1 on the
out-of-fold predictions of a stratified 5-fold CV (seed 2026) instead of
being 0.5.

## Result

Same 5 folds and threshold rule for every row (F1, mean ± std over folds):

| Model | F1 |
|---|---|
| `ridge_regression` (lambda = 1e-4), raw standardized columns | 0.410 ± 0.004 |
| `reg_logistic_regression` (GD, 300 iterations), raw standardized columns | 0.411 ± 0.003 |
| Newton logistic, encoded features | 0.432 ± 0.003 |
| Newton logistic, encoded features + interactions (submitted) | 0.434 ± 0.004 |

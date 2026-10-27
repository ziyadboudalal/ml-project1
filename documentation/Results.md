# Experiment Results

One row per run. Compare using the same metric and validation split/seed.

Best run: **TBD**

## Ismail

| Run | Approach / features | Hyperparameters | Split / seed | Metric | Validation score | AIcrowd score | Code / commit | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISM-01 | `ridge_regression` on raw columns (mean-imputed, standardized), labels in {-1, 1} | λ = 1e-4 | 5-fold stratified / 2026 | F1 | 0.410 ± 0.004 | – | `documentation/figures.py` | Baseline. Treats survey codes (7, 9, 77, 99…) as numbers. |
| ISM-02 | `reg_logistic_regression` (GD) on the same raw columns | λ = 1e-4, γ = 0.15, 300 iters | 5-fold stratified / 2026 | F1 | 0.411 ± 0.003 | – | `documentation/figures.py` | Baseline. Same features as ISM-01: the loss barely matters, features do. |
| ISM-03 | Newton logistic on encoded features: one-hot (≤ 60 values, survey codes kept as levels), clipped + standardized + squared continuous columns, missing indicators | λ = 3e-4 | 5-fold stratified / 2026 | F1 | 0.432 ± 0.003 | – | `documentation/figures.py` | +0.022 over ISM-02: the main gain of the project. ~1 230 columns. |
| ISM-04 | ISM-03 + pairwise products of the 20 encoded columns most correlated with y | λ = 3e-4, 20 interaction columns | 5-fold stratified / 2026 | F1 | 0.434 ± 0.004 | **0.442** (acc. 0.873, sub. 337993) | `submissions/ismail.py` | **Submitted.** +0.002 over ISM-03, within one std. ~1 410 columns. |
| ISM-05 | ISM-04 with 30 interaction columns | λ = 3e-4 | – | – | not completed | – | – | Dropped: ~430 products, 2.2 GB matrix, swapped on 8 GB RAM, Newton diverged (singular Hessian). |
| ISM-06 | Gradient-boosted trees in NumPy | – | single holdout | F1 | 0.445 | – | not in repo | Dropped: harder to explain in 2 pages, gain ≈ +0.01 not measured on the same folds. |

### Notes for the report (Ismail)

- **Threshold.** Positives are 8.8% of rows. With a 0.5 threshold almost
  nothing is predicted positive, so F1 is very low. The threshold is chosen
  to maximize F1 on out-of-fold probabilities (≈ 0.20), see
  `figures/threshold.png`.
- **Feature encoding is what matters.** Ridge and logistic GD on the same raw
  features score the same (0.410 vs 0.411). One-hot encoding of the
  categorical answers brings +0.022 (`figures/ablation.png`). Reason: BRFSS
  answers are codes, e.g. GENHLTH 1–5 or 7 = "don't know", so a single
  linear weight on the raw value makes no sense.
- **Interactions** add +0.002, within noise. We keep them in the submission
  but they are not a key claim.
- **Regularization.** Effect of λ in `figures/lambda.png`.
- **Newton vs GD.** With ~1 400 columns, GD needs many iterations and a tuned
  step size. Newton converges in about 10 iterations. The Hessian is
  accumulated by blocks of rows, and a step that increases the loss is
  halved.
- **CV vs test.** 0.434 in CV vs 0.442 on AIcrowd: no overfitting of the validation.
- **Interpretation.** `figures/weights.png`: female sex, very good general
  health and no stroke/COPD lower the risk; poor general health, kidney
  disease and cholesterol checks raise it. Weights of one-hot levels are
  relative to the reference level of the same question.
- **Data.** `figures/missing_values.png`: many features are missing in more
  than half of the rows, hence the missing indicators.
- **Leakage.** Encoder statistics, interaction selection and weights are
  refitted inside each fold. Only the threshold uses all out-of-fold
  predictions.

## Nour

| Run | Approach / features | Hyperparameters | Split / seed | Metric | Validation score | AIcrowd score | Code / commit | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NOU-01 | | | | | | | | |
| NOU-02 | | | | | | | | |
| NOU-03 | | | | | | | | |

## Ziyad

| Run | Approach / features | Hyperparameters | Split / seed | Metric | Validation score | AIcrowd score | Code / commit | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ZIY-01 | | | | | | | | |
| ZIY-02 | | | | | | | | |
| ZIY-03 | | | | | | | | |

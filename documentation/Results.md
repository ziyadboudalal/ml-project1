# CouscousAI - our experiment results

We evaluated our candidate methods on one common protocol: three shuffled,
non-stratified folds, seed 42. Preprocessing, feature selection and model
weights are fitted within each training fold. All figures and the final report
are generated from these recomputed results.

## Verified common-fold comparison

| Our model | F1 mean | Population SD | Threshold |
|---|---:|---:|---:|
| Ridge on standardized raw features | 0.409160 | 0.002113 | 0.220 |
| Newton logistic, encoded features | 0.430164 | 0.002802 | 0.195 |
| Newton logistic + selected interactions | 0.431102 | 0.003135 | 0.205 |
| Gradient-boosted trees | 0.442141 | 0.003480 | 0.220 |
| Our delivered 70/30 tree-logistic blend | 0.443602 | 0.003610 | 0.225 |

Full-precision values, per-fold confusion matrices and protocol are stored in
`../artifacts/validation/results.json`. The retained tree counts are 183, 171
and 216. Their mean multiplied by 1.1 gives 209 for the full-data refit.

We found that interactions bring a small change (+0.000938 F1), whereas the
boosted-tree pipeline improves more over the logistic interaction variant
(+0.011039). Blending adds +0.001460 over trees alone. These are paired
comparisons on identical folds; small differences remain within fold
variability. The tree/logistic comparison also changes the representation,
so it does not isolate the optimizer alone.

## Our final configuration

Our logistic encoder uses at most 30 distinct observed values for categorical
encoding, with a special rule for state; L2 is 1e-4 including the bias. We add
squares of 10 continuous features and products of 25 selected encoded columns.
Our trees use depth 4, learning rate 0.1 and half of the rows/features per tree.
The delivered model uses 209 trees, tree probability weight
0.7000000000000001 and a classification threshold of 0.225.

We retained this configuration and regenerated its figures. Earlier
experiments using a 60-category cutoff, five folds or 70/15/15 splits are not
mixed into this comparison. The category cutoff is a recorded design choice,
not a claim that 30 is universally optimal.

## Interpretation and limitations

We selected model settings, blend weight and threshold on development
out-of-fold scores. Therefore these are development estimates, not results on
a final untouched holdout. Fold SD is not a confidence interval. Similar
local and leaderboard scores do not prove absence of overfitting.

Our earlier least-squares experiment was numerically unstable on collinear
features. We corrected the required implementation to form float64 normal
equations and handle rank deficiency. We do not use the unstable historical
result as evidence of poor least-squares performance.

## Prediction and competition provenance

We retrained the delivered configuration on all 328,135 labeled rows and
produced 109,379 test predictions, including 13,390 positives. The exact CSV
SHA-256 is:

    4986920c9485b722c1b2992667dba8aa43f9f0d2122df8572229bcc412e775ce

Our earlier experiment log reports AIcrowd F1 0.455. The original uploaded CSV
was unavailable for exact comparison. We therefore distinguish that historical
score from the verified local results and do not certify an AIcrowd score for
the regenerated file before a new upload. The saved weights and CSV allow the
delivered predictions to be reproduced exactly.

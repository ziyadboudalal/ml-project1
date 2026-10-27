"""L2-regularized logistic regression (Newton's method) on encoded survey answers.


Runs stratified 5-fold CV to pick the decision threshold, refits on all
labeled data and writes model.json and submission.csv to submissions/ismail/.
"""

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import create_csv_submission  # noqa: E402

SEED = 2026
FOLDS = 5
LAMBDA = 3e-4
INTERACTIONS = 20
CHUNK = 16384  # rows processed at once to bound memory

# Identifiers, interview dates, state and survey weights.
DROP = {
    "_STATE",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "FMONTH",
    "SEQNO",
    "_PSU",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALCOR",
    "_LLCPWT",
}
# BRFSS codes for "don't know", "none", "refused".
SURVEY_CODES = [7, 8, 9, 77, 88, 99, 777, 888, 999, 7777, 8888, 9999]


def load_features(path):
    with open(path, newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        blocks, ids, rows = [], [], []
        for row in reader:
            ids.append(int(row[0]))
            rows.append([float(v) if v else np.nan for v in row[1:]])
            if len(rows) == 4096:
                blocks.append(np.asarray(rows, dtype=np.float32))
                rows = []
        if rows:
            blocks.append(np.asarray(rows, dtype=np.float32))
    return np.concatenate(blocks), np.asarray(ids), header[1:]


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def stratified_folds(y, k=FOLDS, seed=SEED):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == label))
        for fold, part in zip(folds, np.array_split(indices, k)):
            fold.extend(part)
    return [np.sort(fold) for fold in folds]


def f1_score(y, probability, threshold):
    prediction = probability >= threshold
    tp = np.sum(prediction & (y == 1))
    return 2 * tp / max(np.sum(prediction) + np.sum(y == 1), 1)


def best_threshold(y, probability):
    """Threshold maximizing F1, searched over all distinct scores."""
    order = np.argsort(-probability, kind="stable")
    scores, labels = probability[order], y[order]
    end = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    tp = np.cumsum(labels)[end]
    f1 = 2 * tp / (end + 1 + np.sum(y))
    return float(scores[end[np.argmax(f1)]])


class FeatureEncoder:
    """Encode raw survey answers; all statistics come from the rows passed to fit().

    - Columns with at most `max_levels` values are one-hot encoded; survey
      codes keep their own level and rare levels join the reference level.
    - Other columns are clipped, standardized and expanded with a square;
      frequent survey codes become indicators.
    - Columns with missing values get a missing indicator.
    - Pairwise products of the `interactions` columns most correlated with y.
    """

    def __init__(self, min_fraction=0.002, max_levels=60, interactions=INTERACTIONS):
        self.min_fraction = min_fraction
        self.max_levels = max_levels
        self.interactions = interactions

    def fit(self, x, y, names):
        n = len(x)
        self.names, self.specs = list(names), []
        for j, name in enumerate(names):
            column = x[:, j]
            valid = column[np.isfinite(column)]
            if name in DROP or len(valid) < 0.01 * n or np.ptp(valid) == 0:
                continue
            levels, counts = np.unique(valid, return_counts=True)
            if len(levels) <= self.max_levels:
                kept = levels[counts >= self.min_fraction * n]
                if len(kept) == len(levels):
                    kept = kept[
                        1:
                    ]  # reference level, avoids collinearity with the bias
                spec = dict(col=j, kind="cat", levels=kept.tolist())
            else:
                codes = [
                    float(code)
                    for code in SURVEY_CODES
                    if np.mean(valid == code) > self.min_fraction
                    and code > np.quantile(valid, 0.9)
                ]
                values = valid[~np.isin(valid, codes)]
                low, high = np.quantile(values, [0.01, 0.99])
                values = np.clip(values, low, high)
                spec = dict(
                    col=j,
                    kind="num",
                    low=float(low),
                    high=float(high),
                    mean=float(values.mean()),
                    std=float(values.std()) or 1.0,
                    codes=codes,
                )
            spec["missing"] = bool(np.mean(~np.isfinite(column)) > self.min_fraction)
            self.specs.append(spec)
        self.pairs = self._select_pairs(x, y) if self.interactions else []
        return self

    def _base(self, x):
        """Encoded columns without interactions, and the raw column each comes from."""
        columns, sources = [np.ones(len(x), dtype=np.float32)], [-1]
        for spec in self.specs:
            values = x[:, spec["col"]]
            missing = ~np.isfinite(values)
            if spec["kind"] == "cat":
                new = [values == level for level in spec["levels"]]
            else:
                coded = missing | np.isin(values, spec["codes"])
                clean = np.clip(
                    np.where(coded, spec["mean"], values), spec["low"], spec["high"]
                )
                z = (clean - spec["mean"]) / spec["std"]
                new = [z, z * z - 1] + [values == code for code in spec["codes"]]
            if spec["missing"]:
                new.append(missing)
            columns += new
            sources += [spec["col"]] * len(new)
        return np.column_stack(columns).astype(np.float32), np.array(sources)

    def _select_pairs(self, x, y):
        rows = np.arange(0, len(x), 4)
        base, sources = self._base(x[rows])
        target = y[rows] - y[rows].mean()
        centered = base - base.mean(axis=0)
        scale = np.sqrt((centered**2).mean(axis=0) * (target**2).mean()) + 1e-12
        correlation = np.abs(centered.T @ target / len(rows)) / scale
        correlation[0] = 0.0
        top = np.argsort(-correlation)[: self.interactions]
        # Skip pairs from the same question and products that are almost always 0.
        return [
            (int(a), int(b))
            for i, a in enumerate(top)
            for b in top[i + 1 :]
            if sources[a] != sources[b]
            and np.mean(base[:, a] * base[:, b] != 0) > self.min_fraction
        ]

    def transform_block(self, x):
        base, _ = self._base(x)
        products = [base[:, a] * base[:, b] for a, b in self.pairs]
        return np.column_stack([base] + products) if products else base

    def transform(self, x):
        result = np.empty((len(x), self.transform_block(x[:1]).shape[1]), np.float32)
        for start in range(0, len(x), CHUNK):
            result[start : start + CHUNK] = self.transform_block(
                x[start : start + CHUNK]
            )
        return result

    def to_dict(self):
        return dict(names=self.names, specs=self.specs, pairs=self.pairs)

    @classmethod
    def from_dict(cls, state):
        encoder = cls()
        encoder.names, encoder.specs = state["names"], state["specs"]
        encoder.pairs = [tuple(pair) for pair in state["pairs"]]
        return encoder


def newton_logistic(tx, rows, y, lambda_, max_iters=20, tol=1e-5):
    """Minimize mean log-loss + lambda_ * ||w||^2 (bias not penalized) on tx[rows].

    Gradient and Hessian are accumulated over row blocks. A step that
    increases the loss is halved.
    """
    n, d = len(rows), tx.shape[1]
    penalty = np.full(d, 2 * lambda_)
    penalty[0] = 0.0
    w = np.zeros(d)
    w[0] = np.log(y[rows].mean() / (1 - y[rows].mean()))
    previous, step = np.inf, np.zeros(d)
    for _ in range(max_iters):
        loss = 0.5 * np.sum(penalty * w * w)
        gradient, hessian = penalty * w, np.diag(penalty + 1e-8)
        for start in range(0, n, CHUNK):
            block = rows[start : start + CHUNK]
            x_block = tx[block]
            z = (x_block @ w.astype(np.float32)).astype(float)
            p = sigmoid(z)
            loss += np.sum(np.logaddexp(0.0, z) - y[block] * z) / n
            gradient += x_block.T @ (p - y[block]).astype(np.float32) / n
            weights = (p * (1 - p)).astype(np.float32)
            hessian += (x_block.T @ (x_block * weights[:, None])).astype(float) / n
        if loss > previous:
            step /= 2
            w += step
        else:
            previous = loss
            step = np.linalg.solve(hessian, gradient)
            w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w


def predict_proba(tx, rows, w):
    w = w.astype(np.float32)
    scores = [tx[rows[i : i + CHUNK]] @ w for i in range(0, len(rows), CHUNK)]
    return sigmoid(np.concatenate(scores).astype(float))


def main():
    x, train_ids, names = load_features(ROOT / "dataset" / "x_train.csv")
    labels = np.loadtxt(
        ROOT / "dataset" / "y_train.csv", delimiter=",", skiprows=1, dtype=int
    )
    if not np.array_equal(labels[:, 0], train_ids):
        raise ValueError("Training feature/label IDs differ")
    y = (labels[:, 1] == 1).astype(float)

    # Out-of-fold probabilities; the encoder is refitted inside every fold.
    folds = stratified_folds(y)
    out_of_fold = np.zeros(len(y))
    for k, valid in enumerate(folds):
        train = np.sort(np.concatenate([f for i, f in enumerate(folds) if i != k]))
        encoder = FeatureEncoder().fit(x[train], y[train], names)
        tx = encoder.transform(x)
        w = newton_logistic(tx, train, y, LAMBDA)
        out_of_fold[valid] = predict_proba(tx, valid, w)
    threshold = best_threshold(y, out_of_fold)
    fold_f1 = [f1_score(y[f], out_of_fold[f], threshold) for f in folds]
    print(f"CV F1 = {np.mean(fold_f1):.4f} +/- {np.std(fold_f1):.4f}", flush=True)

    encoder = FeatureEncoder().fit(x, y, names)
    tx = encoder.transform(x)
    w = newton_logistic(tx, np.arange(len(y)), y, LAMBDA)
    del tx
    output = ROOT / "submissions" / "ismail"
    output.mkdir(parents=True, exist_ok=True)
    model = dict(encoder=encoder.to_dict(), weights=w.tolist(), threshold=threshold)
    (output / "model.json").write_text(json.dumps(model) + "\n")
    reproduce(ROOT / "dataset", output)


def reproduce(data, output):
    """Write submission.csv from the saved model.json, without training."""
    model = json.loads((output / "model.json").read_text())
    encoder = FeatureEncoder.from_dict(model["encoder"])
    x, ids, names = load_features(data / "x_test.csv")
    if names != encoder.names:
        raise ValueError("Test columns differ from training")
    w = np.asarray(model["weights"], dtype=np.float32)
    scores = [
        encoder.transform_block(x[i : i + CHUNK]) @ w for i in range(0, len(x), CHUNK)
    ]
    probability = sigmoid(np.concatenate(scores).astype(float))
    prediction = np.where(probability >= model["threshold"], 1, -1)
    create_csv_submission(ids, prediction, output / "submission.csv")


if __name__ == "__main__":
    main()


"""CS-433 Project 1 — independent NumPy-only classification experiments.

Run from repository root:
    python submissions/ziyad.py --mode validate
    python submissions/ziyad.py --mode validate --features clinical
    python submissions/ziyad.py --mode validate --features expanded_no_interactions
    python submissions/ziyad.py --mode submit --features expanded_no_interactions

Only NumPy and the Python standard library are used. Do not tune against AIcrowd.
"""

import argparse
import csv
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
SEED = 42

# Avoid IDs / survey sampling weights / interview bookkeeping.
EXCLUDE = {
    "IDATE", "SEQNO", "_PSU", "_STSTR", "_STRWT", "_RAWRAKE",
    "_WT2RAKE", "_CLLCPWT", "_LLCPWT", "DISPCODE", "CTELENUM",
    "STATERES", "CELLFON2", "CTELNUM1", "CSTATE", "MSCODE",
}

# Numerical features treated as continuous.
CONTINUOUS = {
    "PHYSHLTH", "MENTHLTH", "POORHLTH", "_BMI5", "_AGE80",
    "HTM4", "WTKG3", "_DRNKWEK", "_FRUTSUM", "_VEGESUM",
    "MAXVO2_", "FC60_", "PA1MIN_", "PA1VIGM_",
}

DAYS = {"PHYSHLTH", "MENTHLTH", "POORHLTH"}

# Interpretable clinical feature baseline.
CLINICAL = {
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH", "HLTHPLN1",
    "PERSDOC2", "MEDCOST", "CHECKUP1", "BPHIGH4", "BPMEDS",
    "BLOODCHO", "CHOLCHK", "TOLDHI2", "CVDSTRK3", "ASTHMA3",
    "CHCCOPD1", "HAVARTH3", "ADDEPEV2", "CHCKIDNY", "DIABETE3",
    "SEX", "MARITAL", "EDUCA", "EMPLOY1", "INCOME2", "SMOKE100",
    "SMOKDAY2", "_SMOKER3", "_RFSMOK3", "DRNKANY5", "_RFBING5",
    "EXERANY2", "_TOTINDA", "_PACAT1", "_BMI5", "_BMI5CAT",
    "_AGE80", "_AGEG5YR", "_RFHYPE5", "_RFCHOL", "_RFHLTH",
    "_CHOLCHK", "_INCOMG", "_EDUCAG", "_RFBMI5", "_AGE65YR",
    "QLACTLM2", "DIFFWALK", "DIFFDRES", "DIFFALON", "BLIND",
    "DECIDE", "_RACEGR3", "_FRUITEX", "_VEGETEX", "_DRNKWEK",
}


def read_features(path):
    """Read the dataset while preserving feature names and sample IDs."""
    with open(path, "r", newline="") as f:
        names = next(csv.reader(f))
        ncols = len(names)

        rows = sum(1 for _ in f)
        f.seek(0)
        f.readline()

        x = np.empty((rows, ncols - 1), dtype=np.float32)
        ids = np.empty(rows, dtype=np.int64)

        for i, raw in enumerate(f):
            line = raw.rstrip("\r\n")

            # Replace empty CSV fields with NaN.
            while ",," in line:
                line = line.replace(",,", ",nan,")

            if line.endswith(","):
                line += "nan"

            a = np.fromstring(line, sep=",", dtype=np.float64)

            if len(a) != ncols:
                raise ValueError(
                    f"Invalid CSV row {i + 2}: {len(a)} != {ncols}"
                )

            ids[i] = int(a[0])
            x[i] = a[1:]

    return x, ids, names[1:]


def load_labels(path):
    """Load labels and convert {-1, 1} to {0, 1}."""
    y = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=1,
        dtype=np.int8,
    )

    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Unexpected labels: expected -1 / +1")

    return (y == 1).astype(np.float32)


def split_stratified(y, fraction=0.2, seed=SEED):
    """Create a reproducible stratified train-validation split."""
    rng = np.random.default_rng(seed)

    train = []
    valid = []

    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)

        n = int(round(len(idx) * fraction))

        valid.append(idx[:n])
        train.append(idx[n:])

    tr = np.concatenate(train)
    va = np.concatenate(valid)

    rng.shuffle(tr)
    rng.shuffle(va)

    return tr, va


def clean_numeric(v, name):
    """Handle missing values and special numeric codes."""
    v = v.copy()
    v[~np.isfinite(v)] = np.nan

    if name in DAYS:
        # BRFSS code 88 means zero days.
        v[v == 88] = 0
        v[np.isin(v, [77, 99])] = np.nan

    if name == "_AGE80":
        v[(v < 18) | (v > 80)] = np.nan

    if name == "_BMI5":
        v[(v < 10) | (v > 90)] = np.nan

    return v


class FeatureEncoder:
    """NumPy-only feature preprocessing and encoding.

    The encoder is fitted only on training data to avoid leakage.
    """

    def __init__(self, feature_mode="expanded"):
        self.feature_mode = feature_mode

    def fit(self, x, names):
        self.specs = []
        self.feature_names = []

        for j, name in enumerate(names):
            if name in EXCLUDE:
                continue

            if (
                self.feature_mode == "clinical"
                and name not in CLINICAL
            ):
                continue

            values = x[:, j]
            finite = np.isfinite(values)
            rate = np.mean(finite)

            min_rate = (
                0.10 if self.feature_mode == "clinical" else 0.30
            )

            if rate < min_rate:
                continue

            if name in CONTINUOUS:
                v = clean_numeric(values, name)
                obs = v[np.isfinite(v)]

                if len(obs) < 500:
                    continue

                lower, upper = np.percentile(obs, [1.0, 99.0])
                obs = np.clip(obs, lower, upper)

                center = float(np.mean(obs))
                scale = float(np.std(obs))

                if scale < 1e-7:
                    continue

                self.specs.append(
                    (
                        j,
                        "num",
                        name,
                        (lower, upper, center, scale),
                    )
                )

                self.feature_names.extend(
                    [name, name + "_missing"]
                )

            else:
                unique, counts = np.unique(
                    values[finite],
                    return_counts=True,
                )

                if len(unique) < 2:
                    continue

                max_levels = 55 if name == "_STATE" else 24

                if len(unique) > max_levels:
                    continue

                min_count = max(
                    40,
                    int(x.shape[0] * 0.0003),
                )

                keep = unique[counts >= min_count]

                if len(keep) < 1:
                    continue

                self.specs.append(
                    (
                        j,
                        "cat",
                        name,
                        keep.astype(np.float32),
                    )
                )

                self.feature_names.extend(
                    [f"{name}={val:g}" for val in keep]
                )

                if rate < 0.9999:
                    self.feature_names.append(
                        name + "_missing"
                    )

        self.add_interactions = (
            self.feature_mode == "expanded"
        )

        if self.add_interactions:
            self.feature_names.extend(
                [
                    "age_x_diabetes",
                    "age_x_highbp",
                    "age_x_stroke",
                    "age_x_walk_difficulty",
                    "age_x_smoking",
                    "diabetes_x_highbp",
                    "age_x_bmi",
                ]
            )

        self.name_to_idx = {
            name: j for j, name in enumerate(names)
        }

        self.output_dim = len(self.feature_names) + 1

        return self

    def transform(self, x):
        """Apply transformations learned on the training set."""
        result = np.empty(
            (len(x), self.output_dim),
            dtype=np.float32,
        )

        # Intercept
        result[:, 0] = 1.0
        col = 1

        for j, kind, name, params in self.specs:
            v = x[:, j]
            valid = np.isfinite(v)

            if kind == "num":
                lower, upper, center, scale = params

                cleaned = clean_numeric(v, name)
                valid_num = np.isfinite(cleaned)

                temp = np.nan_to_num(
                    cleaned,
                    nan=center,
                )

                result[:, col] = np.clip(
                    (
                        np.clip(temp, lower, upper)
                        - center
                    ) / scale,
                    -5,
                    5,
                )

                result[:, col + 1] = ~valid_num
                col += 2

            else:
                for category in params:
                    result[:, col] = (v == category)
                    col += 1

                if name + "_missing" in self.feature_names:
                    result[:, col] = ~valid
                    col += 1

        if self.add_interactions:
            d = self.name_to_idx

            age = (
                x[:, d["_AGE80"]]
                if "_AGE80" in d
                else np.full(len(x), np.nan)
            )

            age = np.clip(
                np.nan_to_num(age, nan=45.0),
                18,
                80,
            )

            age01 = (age - 18) / 62

            def yes(name):
                if name not in d:
                    return np.zeros(
                        len(x),
                        dtype=np.float32,
                    )

                return (
                    x[:, d[name]] == 1
                ).astype(np.float32)

            diab = yes("DIABETE3")
            bp = yes("BPHIGH4")
            stroke = yes("CVDSTRK3")
            walk = yes("DIFFWALK")
            smoke = yes("SMOKE100")

            bmi = (
                x[:, d["_BMI5"]]
                if "_BMI5" in d
                else np.full(len(x), 27.0)
            )

            bmi = np.clip(
                np.nan_to_num(bmi, nan=27.0),
                10,
                70,
            )

            interactions = (
                age01 * diab,
                age01 * bp,
                age01 * stroke,
                age01 * walk,
                age01 * smoke,
                diab * bp,
                age01 * ((bmi - 27) / 15),
            )

            for arr in interactions:
                result[:, col] = arr
                col += 1

        if col != self.output_dim:
            raise RuntimeError(
                f"Feature count mismatch: "
                f"{col} versus {self.output_dim}"
            )

        return result

    def fit_transform(self, x, names):
        return self.fit(x, names).transform(x)


def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def binary_loss(y, p):
    """Mean binary cross-entropy."""
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)

    return float(
        np.mean(
            -y * np.log(p)
            - (1 - y) * np.log(1 - p)
        )
    )


def train_adam(
    x,
    y,
    xv=None,
    yv=None,
    epochs=14,
    batch_size=8192,
    learning_rate=0.03,
    l2=0.001,
    seed=SEED,
    verbose=True,
):
    """Train L2-regularized logistic regression with Adam."""
    rng = np.random.default_rng(seed)

    weights = np.zeros(
        x.shape[1],
        dtype=np.float32,
    )

    # Initialize intercept using positive class prevalence.
    prevalence = float(np.mean(y))

    weights[0] = np.log(
        prevalence / (1.0 - prevalence)
    )

    m = np.zeros_like(weights)
    v = np.zeros_like(weights)

    t = 0
    best = np.inf

    best_weights = weights.copy()
    best_epoch = 1
    no_improve = 0

    for epoch in range(epochs):
        ids = rng.permutation(len(y))

        for start in range(0, len(y), batch_size):
            batch = ids[start:start + batch_size]

            xb = x[batch]
            yb = y[batch]

            p = sigmoid(xb @ weights)

            g = (
                xb.T @ (p - yb)
            ) / len(yb)

            # Do not regularize intercept.
            g[1:] += 2 * l2 * weights[1:]

            t += 1

            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * (g * g)

            m_hat = m / (1 - 0.9**t)
            v_hat = v / (1 - 0.999**t)

            weights -= (
                learning_rate
                * m_hat
                / (np.sqrt(v_hat) + 1e-8)
            )

        if xv is not None:
            loss = binary_loss(
                yv,
                sigmoid(xv @ weights),
            )

            if verbose:
                print(
                    f"  epoch {epoch + 1:02d}/{epochs}"
                    f"  validation logloss={loss:.5f}",
                    flush=True,
                )

            if loss < best - 0.0001:
                best = loss
                best_weights = weights.copy()
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= 4:
                break

        elif verbose:
            print(
                f"  epoch {epoch + 1:02d}/{epochs} completed",
                flush=True,
            )

    if xv is not None:
        return best_weights, best_epoch

    return weights, epochs


def metrics(y, pred):
    """Calculate accuracy, precision, recall, and F1-score."""
    truth = y.astype(bool)
    pred = pred.astype(bool)

    tp = int(np.count_nonzero(truth & pred))
    fp = int(np.count_nonzero(~truth & pred))
    fn = int(np.count_nonzero(truth & ~pred))

    accuracy = np.mean(truth == pred)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if tp
        else 0.0
    )

    return accuracy, precision, recall, f1


def best_threshold(y, prob, metric="f1"):
    """Select optimal classification threshold on validation data."""
    grid = np.arange(0.08, 0.601, 0.005)

    rows = [
        (
            float(t),
            metrics(y, prob >= t),
        )
        for t in grid
    ]

    metric_index = (
        3 if metric == "f1" else 0
    )

    selected = max(
        rows,
        key=lambda e: e[1][metric_index],
    )

    return selected[0], selected[1]


def print_score(label, y, probability, threshold):
    """Print classification metrics."""
    accuracy, precision, recall, f1 = metrics(
        y,
        probability >= threshold,
    )

    print(
        f"{label:21s} "
        f"threshold={threshold:.3f}  "
        f"accuracy={accuracy:.4f} "
        f"precision={precision:.4f} "
        f"recall={recall:.4f} "
        f"F1={f1:.4f}",
        flush=True,
    )


def run_validation(x, y, names, args):
    """Train and evaluate on a stratified validation set."""
    train_ids, valid_ids = split_stratified(y)

    print(
        f"Train={len(train_ids):,}  "
        f"validation={len(valid_ids):,}  "
        f"positive={np.mean(y):.2%}",
        flush=True,
    )

    base = np.full(len(valid_ids), -1)

    bacc, bp, br, bf = metrics(
        y[valid_ids],
        base == 1,
    )

    print(
        f"Always-negative: "
        f"accuracy={bacc:.4f}, F1={bf:.4f}",
        flush=True,
    )

    encoder = FeatureEncoder(
        args.features
    ).fit(x[train_ids], names)

    print(
        f"Features: {encoder.output_dim} columns "
        f"({len(encoder.specs)} raw variables)",
        flush=True,
    )

    tr = encoder.transform(x[train_ids])
    va = encoder.transform(x[valid_ids])

    print(
        "Training regularized logistic regression...",
        flush=True,
    )

    w, selected_epochs = train_adam(
        tr,
        y[train_ids],
        va,
        y[valid_ids],
        epochs=args.epochs,
        l2=args.l2,
    )

    print(
        f"Selected epoch count for refitting: "
        f"{selected_epochs}",
        flush=True,
    )

    prob = sigmoid(va @ w)

    print_score(
        "Default cutoff",
        y[valid_ids],
        prob,
        0.5,
    )

    for key in ("f1", "accuracy"):
        threshold, _ = best_threshold(
            y[valid_ids],
            prob,
            key,
        )

        print_score(
            "Best validation " + key,
            y[valid_ids],
            prob,
            threshold,
        )

    print(
        "IMPORTANT: validation scores are estimates, "
        "not AIcrowd test scores.",
        flush=True,
    )


def run_submission(x, y, names, args):
    """Train final model and produce AIcrowd submission."""
    tr_ids, va_ids = split_stratified(y)

    # First, select threshold using validation.
    encoder = FeatureEncoder(
        args.features
    ).fit(x[tr_ids], names)

    tr = encoder.transform(x[tr_ids])
    va = encoder.transform(x[va_ids])

    print(
        f"Fit holdout model "
        f"({encoder.output_dim} engineered columns)...",
        flush=True,
    )

    w, selected_epochs = train_adam(
        tr,
        y[tr_ids],
        va,
        y[va_ids],
        epochs=args.epochs,
        l2=args.l2,
    )

    p = sigmoid(va @ w)

    cutoff, _ = best_threshold(
        y[va_ids],
        p,
        args.optimize,
    )

    validation_positive_rate = float(
        np.mean(p >= cutoff)
    )

    print(
        f"Holdout predicted-positive rate: "
        f"{validation_positive_rate:.2%}",
        flush=True,
    )

    print_score(
        "Validation chosen",
        y[va_ids],
        p,
        cutoff,
    )

    del tr, va, w, encoder

    # Refit on all labeled training data.
    print(
        "Refitting same model on all labeled data...",
        flush=True,
    )

    full_encoder = FeatureEncoder(
        args.features
    ).fit(x, names)

    xf = full_encoder.transform(x)

    # Approximately match the number of optimizer updates.
    holdout_updates = (
        selected_epochs
        * int(np.ceil(len(tr_ids) / 8192))
    )

    full_epochs = max(
        1,
        int(
            round(
                holdout_updates
                / np.ceil(len(y) / 8192)
            )
        ),
    )

    print(
        f"Using {full_epochs} full-data epochs "
        f"(~{holdout_updates} best holdout updates)",
        flush=True,
    )

    wf, _ = train_adam(
        xf,
        y,
        epochs=full_epochs,
        l2=args.l2,
        verbose=True,
    )

    full_positive_rate = np.mean(
        sigmoid(xf @ wf) >= cutoff
    )

    print(
        f"Full-data predicted-positive rate at "
        f"holdout cutoff: {full_positive_rate:.2%}",
        flush=True,
    )

    del xf

    # Read and transform competition test data.
    print(
        "Loading unlabeled test CSV...",
        flush=True,
    )

    x_test, ids, test_names = read_features(
        DATA / "x_test.csv"
    )

    if names != test_names:
        raise ValueError(
            "Training and test feature headers differ"
        )

    print(
        "Encoding test features...",
        flush=True,
    )

    xt = full_encoder.transform(x_test)

    print(
        "Generating predictions...",
        flush=True,
    )

    preds = np.where(
        sigmoid(xt @ wf) >= cutoff,
        1,
        -1,
    ).astype(np.int8)

    if not np.all(np.isin(preds, [-1, 1])):
        raise RuntimeError(
            "Invalid prediction labels"
        )

    if len(ids) != 109379:
        raise RuntimeError(
            f"Expected 109379 test rows, "
            f"got {len(ids)}"
        )

    # Write AIcrowd CSV.
    out_path = (
        ROOT
        / "submissions"
        / "ziyad_submission.csv"
    )

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Prediction"])
        writer.writerows(zip(ids, preds))

    print(
        f"Saved {len(preds):,} predictions "
        f"to {out_path}",
        flush=True,
    )

    print(
        f"Positive prediction rate: "
        f"{np.mean(preds == 1):.2%}",
        flush=True,
    )

    print(
        f"Chosen validation cutoff: {cutoff:.3f} "
        f"(optimized for {args.optimize})",
        flush=True,
    )


def main():
    """Parse arguments and launch validation or submission."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["validate", "submit"],
        default="validate",
    )

    parser.add_argument(
        "--features",
        choices=[
            "clinical",
            "expanded_no_interactions",
            "expanded",
        ],
        default="expanded_no_interactions",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--l2",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--optimize",
        choices=["f1", "accuracy"],
        default="f1",
    )

    args = parser.parse_args()

    np.set_printoptions(precision=3)

    t0 = time.monotonic()

    print(
        "Reading training CSV...",
        flush=True,
    )

    x, ids, names = read_features(
        DATA / "x_train.csv"
    )

    y = load_labels(
        DATA / "y_train.csv"
    )

    if len(y) != len(x) or len(x) != len(ids):
        raise ValueError(
            "Training data row mismatch"
        )

    print(
        f"Read {len(y):,} rows, "
        f"{len(names)} features in "
        f"{time.monotonic() - t0:.1f}s",
        flush=True,
    )

    if args.mode == "validate":
        run_validation(
            x,
            y,
            names,
            args,
        )
    else:
        run_submission(
            x,
            y,
            names,
            args,
        )

    print(
        f"Total elapsed time: "
        f"{time.monotonic() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

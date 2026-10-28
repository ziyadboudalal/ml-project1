"""Figures for the report (Ismail's model). Saved as PNG in documentation/figures/.

Usage: python documentation/figures.py  (about 20 minutes, mostly cross-validation)
Every model is evaluated on the same stratified 5 folds as submissions/ismail.py.
"""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from implementations import reg_logistic_regression, ridge_regression  # noqa: E402
from submissions.ismail import (  # noqa: E402
    DROP,
    LAMBDA,
    FeatureEncoder,
    best_threshold,
    f1_score,
    load_features,
    newton_logistic,
    predict_proba,
    sigmoid,
    stratified_folds,
)

OUT = ROOT / "documentation" / "figures"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def standardized_raw(x, rows, names):
    """Baseline features: raw columns, mean-imputed and standardized on `rows`."""
    keep = [j for j, name in enumerate(names) if name not in DROP]
    raw = x[:, keep].astype(float)
    mean = np.nanmean(raw[rows], axis=0)
    raw = np.where(np.isfinite(raw), raw, mean)
    std = raw[rows].std(axis=0)
    usable = std > 0
    raw = (raw[:, usable] - mean[usable]) / std[usable]
    return np.column_stack([np.ones(len(raw)), raw])


def out_of_fold(y, folds, fit_predict):
    scores = np.zeros(len(y))
    for k, valid in enumerate(folds):
        train = np.sort(np.concatenate([f for i, f in enumerate(folds) if i != k]))
        scores[valid] = fit_predict(train, valid)
    return scores


def fold_f1(y, folds, scores):
    threshold = best_threshold(y, scores)
    return [f1_score(y[f], scores[f], threshold) for f in folds]


def feature_labels(encoder):
    """Human-readable name for every encoded column, in model.json order."""
    names, labels = encoder.names, ["bias"]
    for spec in encoder.specs:
        name = names[spec["col"]]
        if spec["kind"] == "cat":
            labels += [f"{name} = {level:g}" for level in spec["levels"]]
        else:
            labels += [name, f"{name}²"] + [f"{name} = {c:g}" for c in spec["codes"]]
        if spec["missing"]:
            labels.append(f"{name} missing")
    return labels + [f"{labels[a]} × {labels[b]}" for a, b in encoder.pairs]


def plot_missing(x):
    missing = np.sort(np.mean(~np.isfinite(x), axis=0))[::-1]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.fill_between(np.arange(len(missing)), missing, color=BLUE, alpha=0.25, lw=0)
    ax.plot(missing, color=BLUE, lw=1.5)
    ax.set_xlabel("Features, sorted by missing rate")
    ax.set_ylabel("Fraction missing")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(missing) - 1)
    half = int(np.sum(missing > 0.5))
    ax.annotate(
        f"{half} of {len(missing)} features\nmissing in > 50% of rows",
        xy=(half, 0.5),
        xytext=(half + 25, 0.72),
        color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6),
    )
    fig.savefig(OUT / "missing_values.png")


def plot_ablation(results):
    labels = [r[0] for r in results][::-1]
    mean = [np.mean(r[1]) for r in results][::-1]
    std = [np.std(r[1]) for r in results][::-1]
    colors = [BLUE if "interactions" in label else GRAY for label in labels]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.barh(labels, mean, xerr=std, color=colors, height=0.6, error_kw=dict(lw=0.8))
    for i, (value, spread) in enumerate(zip(mean, std)):
        ax.text(value + spread + 0.001, i, f"{value:.3f}", va="center", color=INK)
    ax.set_xlim(0.38, 0.45)
    ax.set_xlabel("F1 (5-fold CV, mean ± std)")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "ablation.png")


def plot_lambda(lambdas, scores):
    mean = [np.mean(s) for s in scores]
    std = [np.std(s) for s in scores]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.errorbar(lambdas, mean, yerr=std, color=BLUE, lw=1.5, marker="o", ms=4)
    best = int(np.argmax(mean))
    ax.text(
        0.3, 0.92, f"best: λ = {lambdas[best]:g}", transform=ax.transAxes, color=INK
    )
    ax.set_xscale("log")
    ax.set_xlabel("Regularization λ")
    ax.set_ylabel("F1 (5-fold CV)")
    fig.savefig(OUT / "lambda.png")


def plot_threshold(y, scores):
    chosen = best_threshold(y, scores)
    grid = np.linspace(0.02, 0.8, 157)
    positives = np.sum(y == 1)
    curves = {"F1": [], "Precision": [], "Recall": []}
    for t in grid:
        predicted = scores >= t
        tp = np.sum(predicted & (y == 1))
        curves["F1"].append(2 * tp / (np.sum(predicted) + positives))
        curves["Precision"].append(tp / max(np.sum(predicted), 1))
        curves["Recall"].append(tp / positives)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for (name, values), color in zip(curves.items(), (BLUE, ORANGE, AQUA)):
        ax.plot(grid, values, color=color, lw=1.5, label=name)
    ax.axvline(chosen, color=INK, lw=0.8, ls="--")
    ax.axvline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.text(chosen + 0.01, 0.03, f"chosen {chosen:.2f}", color=INK)
    ax.text(0.51, 0.03, "0.5", color=MUTED)
    ax.set_xlabel("Decision threshold on predicted probability")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.3, 1.0))
    fig.savefig(OUT / "threshold.png")
    return chosen


def plot_weights(top=15):
    model = json.loads((ROOT / "submissions" / "ismail" / "model.json").read_text())
    labels = feature_labels(FeatureEncoder.from_dict(model["encoder"]))
    weights = np.asarray(model["weights"])
    order = np.argsort(-np.abs(weights[1:]))[:top][::-1] + 1
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    colors = [ORANGE if weights[i] > 0 else BLUE for i in order]
    ax.barh([labels[i] for i in order], weights[order], color=colors, height=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Weight (orange: raises risk, blue: lowers risk)")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "weights.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, _, names = load_features(ROOT / "dataset" / "x_train.csv")
    labels = np.loadtxt(
        ROOT / "dataset" / "y_train.csv", delimiter=",", skiprows=1, dtype=int
    )
    y = (labels[:, 1] == 1).astype(float)
    folds = stratified_folds(y)
    plot_missing(x)
    plot_weights()

    def ridge(train, valid):
        tx = standardized_raw(x, train, names)
        w, _ = ridge_regression(2 * y[train] - 1, tx[train], 1e-4)
        return tx[valid] @ w

    def logistic_gd(train, valid):
        tx = standardized_raw(x, train, names)
        w0 = np.zeros(tx.shape[1])
        w, _ = reg_logistic_regression(y[train], tx[train], 1e-4, w0, 300, 0.15)
        return sigmoid(tx[valid] @ w)

    def newton(lambda_, interactions):
        def fit_predict(train, valid):
            encoder = FeatureEncoder(interactions=interactions)
            tx = encoder.fit(x[train], y[train], names).transform(x)
            return predict_proba(tx, valid, newton_logistic(tx, train, y, lambda_))

        return fit_predict

    summary = {}

    def evaluate(name, fit_predict):
        scores = out_of_fold(y, folds, fit_predict)
        f1 = fold_f1(y, folds, scores)
        summary[name] = dict(mean=float(np.mean(f1)), std=float(np.std(f1)))
        print(f"{name:<45} F1 = {np.mean(f1):.4f} +/- {np.std(f1):.4f}", flush=True)
        return scores, f1

    ablation = [
        ("Ridge, raw features", evaluate("Ridge, raw features", ridge)[1]),
        ("Logistic GD, raw features", evaluate("Logistic GD, raw", logistic_gd)[1]),
        ("Newton, encoded features", evaluate("Newton, encoded", newton(LAMBDA, 0))[1]),
    ]
    lambdas, sweep, final_scores = [1e-5, 1e-4, 3e-4, 1e-3], [], None
    for lambda_ in lambdas:
        scores, f1 = evaluate(
            f"Newton + interactions, λ={lambda_:g}", newton(lambda_, 20)
        )
        sweep.append(f1)
        if lambda_ == LAMBDA:
            final_scores = scores
            ablation.append(("Newton, + interactions", f1))
    plot_ablation(ablation)
    plot_lambda(lambdas, sweep)
    summary["threshold"] = plot_threshold(y, final_scores)
    (OUT / "cv_results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

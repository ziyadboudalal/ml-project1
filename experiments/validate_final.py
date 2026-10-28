"""Reproduce our final model comparison on the same three shuffled folds.

All model preprocessing is fitted within the training fold. Early stopping for
the trees uses 10% of that fold, exactly as in the original notebook. Scores
used to select a threshold/blend are development estimates, not untouched test
estimates. Each finished fold is cached and checked against data/code hashes.

    python experiments/validate_final.py
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import final_pipeline as p
from implementations import ridge_regression
from project_io import load_training, sha256

THRESHOLDS = np.linspace(0.02, 0.98, 193)


def folds(n, k=3, seed=42):
    """Match the original notebook's shuffled, non-stratified folds exactly."""
    indices = np.random.RandomState(seed).permutation(n)
    parts = np.array_split(indices, k)
    for i, valid in enumerate(parts):
        yield np.concatenate([part for j, part in enumerate(parts) if i != j]), valid


def metrics(y, scores, threshold):
    prediction = scores >= threshold
    positive = y == 1
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & ~positive))
    fn = int(np.sum(~prediction & positive))
    tn = int(np.sum(~prediction & ~positive))
    return {
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / len(y),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_scores(y, scores, fold_of, threshold=None):
    """Report mean/population SD over folds, using one pooled OOF threshold."""
    if threshold is None:
        values = [metrics(y, scores, t)["f1"] for t in THRESHOLDS]
        threshold = float(THRESHOLDS[np.argmax(values)])
    per_fold = [
        metrics(y[fold_of == i], scores[fold_of == i], threshold) for i in range(3)
    ]
    f1 = [row["f1"] for row in per_fold]
    return {
        "threshold": threshold,
        "mean_f1": float(np.mean(f1)),
        "std_f1": float(np.std(f1)),
        "fold_metrics": per_fold,
        "pooled_metrics": metrics(y, scores, threshold),
    }


def ridge_scores(x_train, y_train, x_valid, names):
    """Simple raw-feature baseline: train-only mean imputation and scaling."""
    keep = [i for i, name in enumerate(names) if name not in p.METADATA_COLUMNS]
    a, b = x_train[:, keep], x_valid[:, keep]
    count = np.isfinite(a).sum(axis=0)
    mean = np.nansum(a, axis=0) / np.maximum(count, 1)
    a = np.where(np.isnan(a), mean, a)
    std = a.std(axis=0)
    useful = (count > 0) & (std > 0)
    a = (a[:, useful] - mean[useful]) / std[useful]
    b = (
        np.where(np.isnan(b[:, useful]), mean[useful], b[:, useful]) - mean[useful]
    ) / std[useful]
    a = np.column_stack([np.ones(len(a)), a])
    b = np.column_stack([np.ones(len(b)), b])
    weights, _ = ridge_regression(y_train, a, 1e-4)
    return b @ weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "validation"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = {
        name: sha256(ROOT / name)
        for name in [
            "final_pipeline.py",
            "implementations.py",
            "project_io.py",
            "experiments/validate_final.py",
        ]
    }
    fingerprints.update(
        {name: sha256(args.data_dir / name) for name in ["x_train.csv", "y_train.csv"]}
    )
    signature = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True).encode()
    ).hexdigest()
    manifest_path = args.output_dir / "protocol.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["signature"] != signature:
            raise ValueError(
                "Cached validation belongs to different data/code. Choose another --output-dir."
            )
    protocol = {
        "signature": signature,
        "sha256": fingerprints,
        "folds": 3,
        "seed": 42,
        "stratified": False,
        "config": p.CONFIG,
        "std_ddof": 0,
        "threshold_candidates": THRESHOLDS.tolist(),
        "interpretation": "Development OOF scores; threshold and model settings selected on these data.",
    }
    manifest_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print("Loading training data for common-fold evaluation...", flush=True)
    x, y, ids, names = load_training(args.data_dir)
    outputs = {
        name: np.empty(len(y))
        for name in ["ridge_raw", "newton", "newton_interactions", "gbdt"]
    }
    fold_of = np.empty(len(y), dtype=int)
    info = []
    for i, (train, valid) in enumerate(folds(len(y))):
        fold_of[valid] = i
        cache = args.output_dir / f"fold_{i + 1}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                np.testing.assert_array_equal(z["ids"], ids[valid])
                for name in outputs:
                    outputs[name][valid] = z[name]
                info.append(json.loads(z["info"].tobytes().decode()))
            print(f"Fold {i + 1}/3: loaded verified cached results", flush=True)
            continue
        start = time.perf_counter()
        print(f"Fold {i + 1}/3: {len(train)} train / {len(valid)} valid", flush=True)
        xt, xv, yt = x[train], x[valid], y[train]
        outputs["ridge_raw"][valid] = ridge_scores(xt, yt, xv, names)
        print("  ridge baseline complete; fitting logistic encoder", flush=True)
        params = p.fit_preprocessor_v3(xt, yt, names, **p.CONFIG["logistic_pk"])
        a, b = p.transform_v3(xt, params), p.transform_v3(xv, params)
        w, _ = p.newton_logistic_regression(
            yt, a, 15, p.CONFIG["logistic_lambda"], p.CONFIG["logistic_pos_weight"]
        )
        outputs["newton"][valid] = p.predict_logistic(b, w)
        print("  Newton without interactions complete", flush=True)
        inter = p.fit_interactions(a, yt, p.CONFIG["logistic_interactions"])
        a, b = p.add_interactions(a, inter), p.add_interactions(b, inter)
        n_columns = a.shape[1]
        w, _ = p.newton_logistic_regression(
            yt, a, 15, p.CONFIG["logistic_lambda"], p.CONFIG["logistic_pos_weight"]
        )
        outputs["newton_interactions"][valid] = p.predict_logistic(b, w)
        del a, b, params, inter, w
        gc.collect()
        print("  Newton with interactions complete; fitting GBDT", flush=True)
        binner = p.fit_binner(xt, names)
        a, b = p.bin_data(xt, binner), p.bin_data(xv, binner)
        permutation = np.random.RandomState(42).permutation(len(yt))
        n_es = len(yt) // 10
        es, fit = np.sort(permutation[:n_es]), np.sort(permutation[n_es:])
        trees = p.gbdt_fit(
            a[:, fit],
            yt[fit],
            a[:, es],
            yt[es],
            n_trees=1000,
            early_stopping=50,
            verbose=True,
            **p.CONFIG["gbdt_params"],
        )
        outputs["gbdt"][valid] = p.gbdt_predict_proba(trees, b)
        fold_info = {
            "fold": i + 1,
            "training_rows": len(train),
            "validation_rows": len(valid),
            "trees": len(trees["trees"]),
            "logistic_columns": n_columns,
            "seconds": time.perf_counter() - start,
        }
        info.append(fold_info)
        payload = {name: values[valid] for name, values in outputs.items()}
        payload.update(
            ids=ids[valid],
            info=np.frombuffer(json.dumps(fold_info).encode(), dtype=np.uint8),
        )
        np.savez_compressed(cache, **payload)
        print(f"Fold {i + 1} saved: {fold_info}", flush=True)
        del xt, xv, yt, a, b, binner, trees
        gc.collect()
    outputs["blend"] = (
        p.CONFIG["blend_weight"] * outputs["gbdt"]
        + (1 - p.CONFIG["blend_weight"]) * outputs["newton_interactions"]
    )
    results = {
        name: evaluate_scores(y, values, fold_of) for name, values in outputs.items()
    }
    results["blend_locked"] = evaluate_scores(
        y, outputs["blend"], fold_of, p.CONFIG["threshold"]
    )
    curves = [
        {
            "weight": float(w),
            **evaluate_scores(
                y,
                w * outputs["gbdt"] + (1 - w) * outputs["newton_interactions"],
                fold_of,
            ),
        }
        for w in np.linspace(0, 1, 11)
    ]
    summary = {
        "protocol": protocol,
        "fold_info": info,
        "results": results,
        "blend_search": curves,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz", ids=ids, y=y, fold=fold_of, **outputs
    )
    for name, result in results.items():
        print(
            f"{name}: F1={result['mean_f1']:.6f} +/- {result['std_f1']:.6f}, t={result['threshold']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

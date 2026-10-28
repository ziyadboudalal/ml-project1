"""CouscousAI final submission: deterministic inference, or full retraining.

python run.py
python run.py --train
python run.py --reference path/to/actually_uploaded.csv
"""

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

import final_pipeline as pipeline
from helpers import create_csv_submission
from project_io import (
    load_features,
    load_model,
    load_training,
    save_model,
    sha256,
    validate_submission,
)

ROOT = Path(__file__).resolve().parent


def competition_score(submission_sha256):
    """Attach the team-confirmed AIcrowd result only to the uploaded CSV bytes."""
    uploaded_sha256 = "4986920c9485b722c1b2992667dba8aa43f9f0d2122df8572229bcc412e775ce"
    if submission_sha256 != uploaded_sha256:
        return {
            "verified_for_this_csv": False,
            "source": "No recorded AIcrowd evaluation for this CSV SHA-256",
        }
    return {
        "submission_id": 338118,
        "submission_date": "2026-10-28",
        "status": "graded",
        "reported_f1": 0.455,
        "reported_accuracy": 0.886,
        "submission_sha256": uploaded_sha256,
        "source": "Team-confirmed upload and supplied AIcrowd result screenshot",
        "verified_for_this_csv": True,
    }


def train(data):
    """Fit our recorded configuration, with unchanged model arithmetic."""
    started = time.perf_counter()
    print("Loading float64 training data...", flush=True)
    x, y, _, names = load_training(data)
    cfg = pipeline.CONFIG
    print(f"Training {len(y)} rows, {len(names)} features", flush=True)
    binner = pipeline.fit_binner(x, names)
    xb = pipeline.bin_data(x, binner)
    print(f"Training GBDT ({cfg['gbdt_trees']} trees)...", flush=True)
    trees = pipeline.gbdt_fit(
        xb, y, n_trees=cfg["gbdt_trees"], verbose=True, **cfg["gbdt_params"]
    )
    del xb
    print("Fitting logistic feature encoder...", flush=True)
    params = pipeline.fit_preprocessor_v3(x, y, names, **cfg["logistic_pk"])
    tx = pipeline.transform_v3(x, params)
    interactions = pipeline.fit_interactions(tx, y, cfg["logistic_interactions"])
    tx = pipeline.add_interactions(tx, interactions)
    del x
    print(f"Training Newton logistic ({tx.shape[1]} columns)...", flush=True)
    weights, loss = pipeline.newton_logistic_regression(
        y, tx, 15, cfg["logistic_lambda"], cfg["logistic_pos_weight"]
    )
    elapsed = time.perf_counter() - started
    print(
        f"Training finished in {elapsed / 60:.1f} min; logistic loss={loss:.6f}",
        flush=True,
    )
    return {
        "schema_version": 1,
        "feature_names": names,
        "config": cfg,
        "binner": binner,
        "gbdt": trees,
        "preprocessor": params,
        "interactions": interactions,
        "weights": weights,
        "training": {
            "rows": len(y),
            "positive_rows": int(y.sum()),
            "logistic_columns": tx.shape[1],
            "logistic_training_loss": float(loss),
            "seconds": elapsed,
            "seed": pipeline.SEED,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "x_train_sha256": sha256(Path(data) / "x_train.csv"),
            "y_train_sha256": sha256(Path(data) / "y_train.csv"),
            "pipeline_sha256": sha256(ROOT / "final_pipeline.py"),
        },
    }


def predict(model, x, names):
    """Reproduce the recorded blend and exact threshold with saved parameters."""
    if names != model["feature_names"]:
        raise ValueError("Feature names/order differ from the fitted model")
    cfg = model["config"]
    xb = pipeline.bin_data(x, model["binner"])
    probability = cfg["blend_weight"] * pipeline.gbdt_predict_proba(model["gbdt"], xb)
    del xb
    tx = pipeline.transform_v3(x, model["preprocessor"])
    tx = pipeline.add_interactions(tx, model["interactions"])
    probability += (1 - cfg["blend_weight"]) * pipeline.predict_logistic(
        tx, model["weights"]
    )
    if not np.isfinite(probability).all():
        raise ValueError("Model produced non-finite probabilities")
    return probability, np.where(probability >= cfg["threshold"], 1, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("MICHD_DATA_PATH", ROOT / "dataset")),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "artifacts" / "final_model.npz"
    )
    parser.add_argument(
        "--train", action="store_true", help="Refit all weights on the labeled dataset"
    )
    parser.add_argument(
        "--reference", type=Path, help="Compare every prediction with an uploaded CSV"
    )
    args = parser.parse_args()
    if args.train:
        model = train(args.data_dir)
        save_model(args.model, model)
        # All prediction paths reload the archive, so inference matches the delivered artifact.
    elif not args.model.exists():
        parser.error("Saved model missing: run python run.py --train first")
    model = load_model(args.model)
    if model["training"]["pipeline_sha256"] != sha256(ROOT / "final_pipeline.py"):
        parser.error(
            "Pipeline source differs from the saved model: restore the matching source or retrain with --train"
        )
    print("Loading test features and generating predictions...", flush=True)
    x, ids, names = load_features(args.data_dir / "x_test.csv")
    probability, predictions = predict(model, x, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(ids, predictions, args.output)
    validate_submission(args.output, ids)
    check = None
    if args.reference:
        reference = validate_submission(args.reference, ids)
        differences = int(np.count_nonzero(reference != predictions))
        check = {
            "reference_sha256": sha256(args.reference),
            "different_predictions": differences,
        }
        if differences:
            raise ValueError(f"Reference differs in {differences} predictions")
    submission_sha256 = sha256(args.output)
    manifest = {
        "model_sha256": sha256(args.model),
        "submission_sha256": submission_sha256,
        "x_test_sha256": sha256(args.data_dir / "x_test.csv"),
        "rows": len(ids),
        "positive_predictions": int((predictions == 1).sum()),
        "positive_fraction": float((predictions == 1).mean()),
        "config": model["config"],
        "training": model["training"],
        "reference_comparison": check,
        "competition_score": competition_score(submission_sha256),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {args.output}: {len(ids)} rows, {(predictions == 1).sum()} positives",
        flush=True,
    )
    print(f"CSV SHA256: {manifest['submission_sha256']}", flush=True)


if __name__ == "__main__":
    main()

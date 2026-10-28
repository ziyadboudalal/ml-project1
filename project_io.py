"""Dataset validation and portable NumPy-only model persistence.

CSV parsing uses float64, like the course loader, without its large temporary
list of all CSV fields. The model archive never uses pickle or object arrays.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    """Hash a file without loading it completely into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_features(path):
    """Read Id + numeric features; preserve row order, missing values and precision."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        n = sum(1 for _ in stream) - 1
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        names = header[1:]
        ids = np.empty(n, dtype=np.int64)
        x = np.empty((n, len(names)), dtype=np.float64)
        for i, row in enumerate(reader):
            if len(row) != len(header):
                raise ValueError(f"Wrong column count in {path.name}, row {i + 2}")
            ids[i] = int(row[0])
            x[i] = [float(v) if v.strip() else np.nan for v in row[1:]]
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Duplicate IDs in {path}")
    if np.isinf(x).any():
        raise ValueError(f"Infinite feature values in {path}")
    return x, ids, names


def load_training(data):
    """Load labeled rows and reject misaligned IDs or unexpected target values."""
    data = Path(data)
    x, ids, names = load_features(data / "x_train.csv")
    labels = np.loadtxt(data / "y_train.csv", delimiter=",", skiprows=1, dtype=np.int64)
    if not np.array_equal(ids, labels[:, 0]):
        raise ValueError("Training features and labels have different IDs/order")
    if not np.isin(labels[:, 1], [-1, 1]).all():
        raise ValueError("Training labels must be -1 or +1")
    return x, (labels[:, 1] == 1).astype(float), ids, names


def save_model(path, model):
    """Store nested model parameters as JSON metadata plus typed NumPy arrays."""
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            name = f"array_{len(arrays):04d}"
            arrays[name] = value
            return {"array": name}
        if isinstance(value, dict):
            return {"dict": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"list": [encode(v) for v in value]}
        if isinstance(value, np.generic):
            return value.item()
        return value

    metadata = json.dumps(encode(model), allow_nan=False).encode("utf-8")
    arrays["metadata"] = np.frombuffer(metadata, dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def load_model(path):
    """Load trusted local model data without allowing pickle execution."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))

        def decode(value):
            if not isinstance(value, dict):
                return value
            if "array" in value:
                return archive[value["array"]].copy()
            if "dict" in value:
                return {decode(k): decode(v) for k, v in value["dict"]}
            return [decode(v) for v in value["list"]]

        return decode(metadata)


def validate_submission(path, expected_ids):
    """Check the CSV schema, all IDs and labels, without silently truncating rows."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        if next(reader) != ["Id", "Prediction"]:
            raise ValueError("Submission header must be Id,Prediction")
        rows = np.asarray([[int(a), int(b)] for a, b in reader], dtype=np.int64)
    if rows.shape != (len(expected_ids), 2):
        raise ValueError("Wrong number of submission rows")
    if not np.array_equal(rows[:, 0], expected_ids):
        raise ValueError("Submission IDs/order differ from x_test.csv")
    if not np.isin(rows[:, 1], [-1, 1]).all():
        raise ValueError("Submission labels must be -1 or +1")
    return rows[:, 1]

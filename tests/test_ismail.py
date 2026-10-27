"""Run with: python -m unittest discover -s tests"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from submissions.ismail import (
    FeatureEncoder,
    best_threshold,
    f1_score,
    newton_logistic,
    reproduce,
    sigmoid,
    stratified_folds,
)


class SubmissionTests(unittest.TestCase):
    def test_reproduce_writes_ids_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "x_test.csv").write_text("Id,feature\n91,0\n7,2\n42,\n")
            spec = dict(col=0, kind="cat", levels=[2.0], missing=True)
            model = dict(
                encoder=dict(names=["feature"], specs=[spec], pairs=[]),
                weights=[-1.0, 2.0, 0.0],
                threshold=0.5,
            )
            (path / "model.json").write_text(json.dumps(model))
            reproduce(path, path)
            with (path / "submission.csv").open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(
                rows, [["Id", "Prediction"], ["91", "-1"], ["7", "1"], ["42", "-1"]]
            )

    def test_folds_are_disjoint_and_stratified(self):
        y = np.r_[np.zeros(900), np.ones(100)]
        folds = stratified_folds(y, k=5)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(1000))
        for fold in folds:
            self.assertAlmostEqual(y[fold].mean(), 0.1)

    def test_best_threshold_matches_exhaustive_search(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        p = np.array([0.8, 0.8, 0.6, 0.6, 0.2, 0.1])
        expected = max(f1_score(y, p, t) for t in np.unique(p))
        self.assertAlmostEqual(f1_score(y, p, best_threshold(y, p)), expected)

    def test_encoder_round_trip_and_unseen_values(self):
        rng = np.random.default_rng(0)
        n = 2000
        x = np.column_stack(
            [rng.normal(size=n), rng.integers(1, 4, size=n), np.ones(n), np.arange(n)]
        ).astype(np.float32)
        x[::10, 0] = np.nan
        y = (x[:, 1] == 3).astype(float)
        encoder = FeatureEncoder(interactions=3).fit(x, y, ["a", "b", "c", "SEQNO"])
        self.assertEqual({spec["col"] for spec in encoder.specs}, {0, 1})
        restored = FeatureEncoder.from_dict(json.loads(json.dumps(encoder.to_dict())))
        unseen = np.array([[np.nan, 7, 1, 0], [1e9, 1, 1, 0]], dtype=np.float32)
        encoded = restored.transform(unseen)
        np.testing.assert_array_equal(encoded, encoder.transform(unseen))
        self.assertTrue(np.isfinite(encoded).all())

    def test_newton_reaches_stationary_point(self):
        rng = np.random.default_rng(1)
        tx = np.column_stack([np.ones(500), rng.normal(size=(500, 3))])
        y = (rng.random(500) < sigmoid(tx @ [-0.5, 1.0, -2.0, 0.5])).astype(float)
        lambda_ = 0.01
        w = newton_logistic(tx.astype(np.float32), np.arange(500), y, lambda_)
        gradient = tx.T @ (sigmoid(tx @ w) - y) / 500 + 2 * lambda_ * np.r_[0, w[1:]]
        self.assertLess(np.abs(gradient).max(), 1e-5)


if __name__ == "__main__":
    unittest.main()

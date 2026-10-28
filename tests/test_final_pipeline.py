"""Numerical, data-integrity and saved-model tests for the delivered pipeline."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import final_pipeline as p
from project_io import (
    load_features,
    load_model,
    load_training,
    save_model,
    validate_submission,
)
from helpers import create_csv_submission
from run import competition_score, predict
from implementations import least_squares


class FinalPipelineTests(unittest.TestCase):
    def test_competition_score_applies_only_to_the_confirmed_csv(self):
        uploaded = "4986920c9485b722c1b2992667dba8aa43f9f0d2122df8572229bcc412e775ce"
        score = competition_score(uploaded)
        self.assertTrue(score["verified_for_this_csv"])
        self.assertEqual(score["submission_id"], 338118)
        self.assertEqual(score["reported_f1"], 0.455)
        self.assertEqual(score["reported_accuracy"], 0.886)
        changed = competition_score("0" * 64)
        self.assertFalse(changed["verified_for_this_csv"])
        self.assertNotIn("reported_f1", changed)
        self.assertNotIn("reported_accuracy", changed)

    def test_least_squares_handles_collinear_float32_features(self):
        col = np.linspace(-1, 1, 100, dtype=np.float32)
        x = np.column_stack([np.ones(100, dtype=np.float32), col, col])
        y = 0.25 + 2 * col.astype(float)
        weights, loss = least_squares(y, x)
        np.testing.assert_allclose(weights, [0.25, 1, 1], atol=1e-10)
        self.assertLess(loss, 1e-20)

    def test_loader_preserves_float64_and_missing_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x_test.csv"
            path.write_text("Id,a,b\n91,0.123456789123,\n7,2,8\n")
            x, ids, names = load_features(path)
            self.assertEqual(x.dtype, np.float64)
            self.assertEqual(x[0, 0], 0.123456789123)
            self.assertTrue(np.isnan(x[0, 1]))
            np.testing.assert_array_equal(ids, [91, 7])
            self.assertEqual(names, ["a", "b"])

    def test_training_loader_rejects_misaligned_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "x_train.csv").write_text("Id,a\n1,2\n2,3\n")
            (path / "y_train.csv").write_text("Id,Prediction\n2,1\n1,-1\n")
            with self.assertRaisesRegex(ValueError, "different IDs"):
                load_training(path)

    def test_special_codes_respect_column_specific_exceptions(self):
        names = ["_RACE", "EMPLOY1", "PHYSHLTH", "WEIGHT2"]
        x = np.array([[7, 7, 88, 9070], [9, 9, 77, 9999]], dtype=float)
        codes = p.learn_heuristic_codes(x, names)
        clean = p.apply_special_codes(x, names, codes)
        np.testing.assert_array_equal(clean[0], [7, 7, 0, 70])
        self.assertTrue(np.isnan(clean[1]).all())

    def test_newton_stationary_point_includes_regularized_intercept(self):
        rng = np.random.RandomState(4)
        tx = np.column_stack([np.ones(600), rng.normal(size=(600, 3))])
        y = (rng.rand(600) < p.sigmoid(tx @ np.array([1.5, 1, -2, 0.5]))).astype(float)
        lam = 0.03
        w, loss = p.newton_logistic_regression(y, tx, 30, lam, tol=1e-12)
        gradient = tx.T @ (p.sigmoid(tx @ w) - y) / len(y) + 2 * lam * w
        self.assertLess(np.max(np.abs(gradient)), 1e-6)
        expected = np.mean(np.logaddexp(0, (1 - 2 * y) * (tx @ w)))
        self.assertAlmostEqual(loss, expected)

    def test_boosting_learns_nonlinearity_and_is_deterministic(self):
        rng = np.random.RandomState(7)
        x = rng.rand(1600, 2)
        y = ((x[:, 0] > 0.5) ^ (x[:, 1] > 0.5)).astype(float)
        x[::13, 1] = np.nan
        binner = p.fit_binner(x[:1200], ["a", "b"], use_codebook=False)
        xb = p.bin_data(x, binner)
        args = dict(
            n_trees=35, max_depth=3, subsample=0.8, colsample=1, min_child_hess=1
        )
        first = p.gbdt_fit(xb[:, :1200], y[:1200], **args)
        second = p.gbdt_fit(xb[:, :1200], y[:1200], **args)
        a = p.gbdt_predict_proba(first, xb[:, 1200:])
        b = p.gbdt_predict_proba(second, xb[:, 1200:])
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.mean((a >= 0.5) == y[1200:]), 0.90)

    def test_saved_blend_preserves_predictions_exactly(self):
        rng = np.random.RandomState(8)
        x = rng.normal(20, 4, (500, 3))
        x[::11, 0] = np.nan
        y = (x[:, 1] > 20).astype(float)
        names = ["a", "b", "c"]
        params = p.fit_preprocessor_v3(x, y, names, use_codebook=False, n_poly=2)
        tx = p.transform_v3(x, params)
        inter = p.fit_interactions(tx, y, 3)
        tx = p.add_interactions(tx, inter)
        w, _ = p.newton_logistic_regression(y, tx)
        binner = p.fit_binner(x, names, use_codebook=False)
        trees = p.gbdt_fit(p.bin_data(x, binner), y, n_trees=8, min_child_hess=1)
        model = dict(
            feature_names=names,
            config=p.CONFIG,
            binner=binner,
            gbdt=trees,
            preprocessor=params,
            interactions=inter,
            weights=w,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            save_model(path, model)
            restored = load_model(path)
            before, labels_before = predict(model, x, names)
            after, labels_after = predict(restored, x, names)
            np.testing.assert_array_equal(before, after)
            np.testing.assert_array_equal(labels_before, labels_after)
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(archive[k].dtype != object for k in archive.files))

    def test_submission_validation_rejects_wrong_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "submission.csv"
            create_csv_submission([91, 7], [1, -1], path)
            np.testing.assert_array_equal(validate_submission(path, [91, 7]), [1, -1])
            with self.assertRaisesRegex(ValueError, "IDs/order"):
                validate_submission(path, [7, 91])


if __name__ == "__main__":
    unittest.main()

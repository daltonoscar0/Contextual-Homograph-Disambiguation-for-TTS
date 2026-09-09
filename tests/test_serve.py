"""Serving tests: the scaler fold, weight loading, model selection."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lede import serve  # noqa: E402


class TestScalerFold(unittest.TestCase):
    """``serve.export`` folds StandardScaler into the logistic layer. If that
    arithmetic is wrong, serving silently disagrees with the reported eval
    numbers, which is the worst kind of bug this repo could ship. Checked on
    synthetic data so the test needs no corpus and no cache."""

    def _pipeline(self, n_classes: int, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        x = rng.normal(size=(120, 16)) * rng.uniform(0.5, 20, size=16)
        y = np.array([f"w{i % n_classes}" for i in range(120)])
        x[y == "w0"] += 1.5
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        pipeline.fit(x, y)
        return pipeline, x

    def _roundtrip(self, n_classes: int):
        pipeline, x = self._pipeline(n_classes)
        path = serve.export(
            f"__test_{n_classes}", {"w": ("model", pipeline, 1.0)}, "final", "balanced"
        )
        try:
            serve.load_probes.cache_clear()
            probes = serve.load_probes(f"__test_{n_classes}")
            expected = pipeline.predict_proba(x)
            for row, reference in zip(x, expected):
                got = probes.probabilities("w", row.astype(np.float32))
                for label, value in zip(pipeline.classes_, reference):
                    self.assertAlmostEqual(got[label], value, places=4)
        finally:
            path.unlink(missing_ok=True)
            serve.load_probes.cache_clear()

    def test_binary_probabilities_match_sklearn(self):
        self._roundtrip(2)

    def test_multiclass_probabilities_match_sklearn(self):
        self._roundtrip(3)

    def test_constant_probes_round_trip(self):
        path = serve.export("__test_const", {"w": ("constant", "only")}, "final", None)
        try:
            serve.load_probes.cache_clear()
            probes = serve.load_probes("__test_const")
            self.assertEqual(probes.probabilities("w", np.zeros(4)), {"only": 1.0})
            self.assertIn("w", probes)
        finally:
            path.unlink(missing_ok=True)
            serve.load_probes.cache_clear()


class TestModelSelection(unittest.TestCase):
    def test_default_is_the_light_encoder(self):
        self.assertEqual(serve.DEFAULT_MODEL, "distilroberta-base")

    def test_env_var_overrides_the_default(self):
        previous = os.environ.get("LEDE_MODEL")
        os.environ["LEDE_MODEL"] = "roberta-large"
        try:
            self.assertEqual(serve.active_model(), "roberta-large")
        finally:
            os.environ.pop("LEDE_MODEL", None)
            if previous is not None:
                os.environ["LEDE_MODEL"] = previous

    def test_missing_weights_name_the_fix(self):
        with self.assertRaises(FileNotFoundError) as caught:
            serve.load_probes("no-such-encoder")
        self.assertIn("make weights", str(caught.exception))


class TestShippedWeights(unittest.TestCase):
    """Whatever weights are checked in must be loadable and well formed."""

    def test_shipped_weights_are_usable(self):
        shipped = sorted(serve.WEIGHTS_DIR.glob("probe_*.npz"))
        if not shipped:
            self.skipTest("no weights checked in")
        for path in shipped:
            model = path.stem.removeprefix("probe_")
            with self.subTest(model=model):
                probes = serve.load_probes(model)
                self.assertEqual(len(probes.manifest), 162)
                self.assertIn(probes.layer_mode, ("final", "last4"))
                self.assertGreater(probes.dimension, 0)
                for array in probes.arrays.values():
                    self.assertTrue(np.isfinite(array).all())

    def test_default_model_weights_are_shipped(self):
        self.assertTrue(
            serve.weights_path(serve.DEFAULT_MODEL).exists(),
            f"{serve.DEFAULT_MODEL} is the serving default and must ship weights",
        )

    def test_pos_fallback_rules_are_shipped(self):
        rules = serve.load_pos_rules()
        self.assertEqual(len(rules["defaults"]), 162)
        self.assertEqual(rules["rules"]["reading"]["PROPN"], "reading_geo")


if __name__ == "__main__":
    unittest.main(verbosity=2)

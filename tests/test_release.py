"""Release checks that require neither source-data downloads nor model training."""

import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from sklearn.covariance import LedoitWolf

from orthogonal_explanations import GaussianCOLA, ordered_cola

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "archive" / "orthogonal_explanations.py"


def archived_definition(name, occurrence=0, **namespace):
    tree = ast.parse(SOURCE.read_text())
    matches = [node for node in tree.body
               if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
    module = ast.Module(body=[matches[occurrence]], type_ignores=[])
    scope = {"np": np, "LedoitWolf": LedoitWolf, **namespace}
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope[name]


class Predictor:
    def __init__(self, linear):
        self.kind = "LR" if linear else "XGB"
        self.weights = np.array([0.007, -0.005, 0.0035, 0.002, -0.0015, 0.001])

    def predict(self, x):
        value = 0.002 + x @ self.weights
        if self.kind != "LR":
            value = value + 0.003 * np.sin(x[:, 0]) + 0.002 * x[:, 2] ** 2
        return value

    def linear_parameters(self):
        return self.weights.copy(), 0.002


class TestGaussianCOLA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(101)
        covariance = 0.7 * np.ones((6, 6)) + 0.3 * np.eye(6)
        self.reference = rng.multivariate_normal(np.zeros(6), covariance, 256).astype(np.float32)
        self.x = rng.multivariate_normal(np.zeros(6), covariance, 19).astype(np.float32)

    def test_matches_archived_experiment_three(self):
        original = archived_definition("GaussianCOLA", occurrence=1,
                                       CFG={"base_draws": 128, "ce_draws": 16, "ce_batch": 7})
        for linear in (True, False):
            with self.subTest(linear=linear):
                model = Predictor(linear)
                old = original(self.reference, model, seed=23)
                new = GaussianCOLA(self.reference, model.predict, seed=23,
                                   baseline_draws=128, batch_size=7,
                                   linear_parameters=model.linear_parameters() if linear else None)
                expected, expected_prediction = old.explain(self.x, seed=24)
                actual, prediction = new.explain(self.x, seed=24)
                assert_allclose(new.base, old.base, rtol=0, atol=1e-14)
                assert_allclose(actual, expected, rtol=0, atol=1e-13)
                assert_allclose(prediction, expected_prediction, rtol=0, atol=1e-14)

    def test_reconstruction_for_both_predictor_types_and_orders(self):
        for linear in (True, False):
            for order in (None, [5, 3, 1, 4, 0, 2]):
                with self.subTest(linear=linear, order=order):
                    model = Predictor(linear)
                    explainer = GaussianCOLA(self.reference, model.predict, order=order,
                                             linear_parameters=model.linear_parameters() if linear else None)
                    a, p = explainer.explain(self.x, seed=3)
                    assert_allclose(a.sum(axis=1) + explainer.base, p, rtol=0, atol=1e-14)

    def test_exact_linear_prefixes_match_gaussian_conditional_means(self):
        model = Predictor(True)
        order = np.array([5, 3, 1, 4, 0, 2])
        explainer = GaussianCOLA(self.reference, model.predict, order=order,
                                 linear_parameters=model.linear_parameters())
        a, _ = explainer.explain(self.x)
        for size in range(1, 6):
            obs, miss = order[:size], order[size:]
            conditional_mean = (explainer.mean[miss] +
                (self.x[:, obs] - explainer.mean[obs]) @
                np.linalg.solve(explainer.cov[np.ix_(obs, obs)], explainer.cov[np.ix_(obs, miss)]))
            expected = 0.002 + self.x[:, obs] @ model.weights[obs] + conditional_mean @ model.weights[miss]
            assert_allclose(explainer.base + a[:, obs].sum(axis=1), expected, atol=1e-14)

    def test_exact_reference_orthogonality(self):
        model = Predictor(True)
        for order in (None, [5, 3, 1, 4, 0, 2]):
            explainer = GaussianCOLA(self.reference, model.predict, order=order,
                                     linear_parameters=model.linear_parameters())
            # Evaluate the affine map on a basis; integrate covariance exactly.
            at_origin, _ = explainer.explain(np.zeros((1, 6), dtype=np.float32))
            at_basis, _ = explainer.explain(np.eye(6, dtype=np.float32))
            transform = at_basis - at_origin
            covariance = transform.T @ explainer.cov @ transform
            off_diagonal = covariance - np.diag(np.diag(covariance))
            assert_allclose(off_diagonal, 0, atol=1e-15)

    def test_fixed_seed_and_batching(self):
        model = Predictor(False)
        one = GaussianCOLA(self.reference, model.predict, seed=19, batch_size=1)
        many = GaussianCOLA(self.reference, model.predict, seed=19, batch_size=512)
        a, p = one.explain(self.x, seed=20)
        b, q = many.explain(self.x, seed=20)
        assert_allclose(a, b, rtol=0, atol=1e-13)
        assert_allclose(p, q, rtol=0, atol=1e-13)

    def test_single_feature(self):
        reference = self.reference[:, :1]
        explainer = GaussianCOLA(reference, lambda x: 2 * x[:, 0], linear_parameters=([2], 0))
        a, p = explainer.explain(self.x[:, :1])
        assert_allclose(a[:, 0], p - explainer.base)

    def test_rejects_invalid_inputs(self):
        model = Predictor(True)
        for kwargs in ({"draws": 3}, {"draws": 0}, {"batch_size": 0},
                       {"order": [0] * 6}, {"order": list(map(float, range(6)))},
                       {"linear_parameters": ([1, 2], 0)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GaussianCOLA(self.reference, model.predict, **kwargs)
        explainer = GaussianCOLA(self.reference, model.predict)
        for x in (np.zeros((2, 5)), np.full((2, 6), np.nan)):
            with self.assertRaises(ValueError):
                explainer.explain(x)
        with self.assertRaises(ValueError):
            GaussianCOLA(self.reference, lambda x: np.zeros((len(x), 2)))


class TestSourceRelease(unittest.TestCase):
    def test_coalition_api_matches_archive(self):
        original = archived_definition("ordered_cola")
        values = np.random.default_rng(3).normal(size=(11, 64))
        for order in (None, [5, 3, 1, 4, 0, 2]):
            a = ordered_cola(values, 6, order)
            assert_allclose(a, original(values, 6, order), rtol=0, atol=0)
            assert_allclose(a.sum(axis=1), values[:, -1] - values[:, 0], atol=1e-14)

    def test_archive_and_generated_files(self):
        spec = importlib.util.spec_from_file_location("extract", ROOT / "tools" / "extract_experiments.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for relative, content in module.generated_files().items():
            self.assertEqual((ROOT / relative).read_bytes(), content.encode())

    def test_experiments_have_no_notebook_installation_calls(self):
        for path in sorted((ROOT / "experiments").glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    self.assertNotEqual(node.id, "get_ipython", path.name)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotEqual(node.func.attr, "check_call", path.name)


if __name__ == "__main__":
    unittest.main()

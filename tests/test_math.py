import unittest
import numpy as np
from llicams.math import (unit, relational_entropy, normalized_relational_entropy, pair_entropy,
                          historical_baseline, sigmoid, project_orthogonal, rank_compress, fit_linear_lora)
from llicams.config import Config


class MathematicsTests(unittest.TestCase):
    def test_original_equation_matches_explicit_double_sum(self):
        rng = np.random.default_rng(8)
        for k in (2, 4, 9):
            p = rng.dirichlet(np.ones(k))
            explicit = -sum(p[i] * p[j] * np.log2(p[i] * p[j]) for i in range(k) for j in range(k) if i != j)
            self.assertAlmostEqual(explicit, relational_entropy(p), places=12)

    def test_known_values_and_bound(self):
        self.assertEqual(relational_entropy([1, 0]), 0)
        self.assertAlmostEqual(relational_entropy([0.5, 0.5]), 1)
        self.assertAlmostEqual(relational_entropy([0.25] * 4), 3)
        self.assertAlmostEqual(pair_entropy([0.25] * 4), np.log2(12))
        for p in np.random.default_rng(9).dirichlet(np.ones(8), size=100):
            self.assertLessEqual(relational_entropy(p), 2 * np.log2(8))
            self.assertGreaterEqual(normalized_relational_entropy(p), 0)

    def test_invalid_probabilities(self):
        for p in ([1], [-0.1, 1.1], [0.2, 0.2], [np.nan, 0]):
            with self.assertRaises(ValueError):
                relational_entropy(p)

    def test_invalid_vectors(self):
        for v in ([0, 0], [np.nan, 1], [[1, 2]]):
            with self.assertRaises(ValueError):
                unit(v)

    def test_sigmoid_and_cold_start(self):
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(sigmoid(-1000), 0)
        self.assertEqual(sigmoid(1000), 1)
        self.assertEqual(historical_baseline(0, 0, 0.6, 50), 0.6)
        self.assertAlmostEqual(historical_baseline(200, 1000, 0.6, 50), 0.2, places=7)

    def test_projection_with_dependent_basis(self):
        a = np.array([[1., 2.], [3., 4.]])
        b = np.array([[2., 0.], [0., 1.]])
        candidate = np.array([[3., 8.], [1., 7.]])
        projected = project_orthogonal(candidate, [a, b, 2 * a])
        self.assertAlmostEqual(float(np.sum(projected * a)), 0, places=10)
        self.assertAlmostEqual(float(np.sum(projected * b)), 0, places=10)

    def test_rank_compression_measures_loss(self):
        matrix = np.diag([4., 2., 1.])
        b, a, residual = rank_compress(matrix, 2)
        self.assertEqual(np.linalg.matrix_rank(b @ a), 2)
        self.assertAlmostEqual(residual, 1)

    def test_linear_lora_really_learns(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(80, 4))
        target = np.diag([0.4, -0.3, 0, 0])
        y = x + x @ target.T
        b, a, losses = fit_linear_lora(x, y, rank=2, steps=1500, learning_rate=0.3)
        self.assertLess(losses[-1], losses[0] * 0.02)
        self.assertLess(np.mean((x + x @ (b @ a).T - y) ** 2), 1e-3)

    def test_config_validation_and_monotone_threshold(self):
        self.assertLess(Config(lambda_li=0).consolidation_threshold, Config(lambda_li=1).consolidation_threshold)
        for kwargs in ({"dimension": 0}, {"lambda_li": 2}, {"epsilon_max": 0},
                       {"selector": "invented"}, {"shadow_threshold": 0.9}):
            with self.assertRaises(ValueError):
                Config(**kwargs)

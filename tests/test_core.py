import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from icboost import check_derivatives, risk
from metrics import strat_auc, net_benefit
from stats_tests import holm, nadeau_bengio


class CoreChecks(unittest.TestCase):
    def test_window_gradient_and_fisher_curvature(self):
        self.assertTrue(check_derivatives())

    def test_window_risk_monotonicity_and_range(self):
        p = risk(np.zeros(40), np.linspace(0.001, 5, 40))
        self.assertTrue(np.all(np.diff(p) > 0))
        self.assertTrue(np.all((p > 0) & (p < 1)))

    def test_stratified_auc_ignores_between_window_order(self):
        times = np.repeat(np.arange(1, 6), 20)
        labels = np.tile([0, 1], 50)
        self.assertAlmostEqual(strat_auc(times, labels, times), 0.5)
        self.assertAlmostEqual(strat_auc(labels, labels, times), 1.0)

    def test_net_benefit_false_positive_cost(self):
        self.assertAlmostEqual(net_benefit([0.9, 0.8, 0.1, 0.1], [1, 0, 1, 0], 0.2), 0.1875)

    def test_holm_preserves_missing_values_and_bounds(self):
        adjusted = holm([0.01, 0.04, np.nan, 0.03])
        np.testing.assert_allclose(adjusted[[0, 1, 3]], [0.03, 0.06, 0.06])
        self.assertTrue(np.isnan(adjusted[2]))

    def test_overlap_correction_reduces_test_statistic(self):
        diff = [0.01, 0.02, -0.005, 0.03, 0.015]
        _, t_small, _ = nadeau_bengio(diff, [100], [10])
        _, t_large, _ = nadeau_bengio(diff, [100], [50])
        self.assertGreater(abs(t_small), abs(t_large))


if __name__ == "__main__":
    unittest.main()

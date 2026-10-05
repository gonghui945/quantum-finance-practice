"""Synthetic independent checks of clustered AUC and calendar blocks."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qml_uncertainty import calendar_blocks, draw_week_counts, auc_pair_matrix, weighted_auc


class UncertaintyTests(unittest.TestCase):
    def test_blocks_do_not_cross_calendar_gaps(self):
        weeks = pd.date_range("2024-01-01", periods=10, freq="W-MON").delete([4, 5])
        blocks = calendar_blocks(weeks, 4)
        np.testing.assert_array_equal(blocks, [[0, 1, 2, 3], [4, 5, 6, 7]])
        counts, _ = draw_week_counts(weeks, 4, replicates=25)
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(25, len(weeks)))
        self.assertTrue(np.all(counts >= 0))

    def test_cluster_auc_matches_explicit_row_resampling(self):
        groups = [pd.DataFrame({"target": [0, 1, 0, 1], "score": scores})
                  for scores in [[0, .5, .5, 1], [.2, .1, .8, .7], [0, 1, 0, 1]]]
        counts = np.array([[1, 1, 1], [2, 0, 1], [0, 3, 0], [0, 0, 3]])
        calculated = weighted_auc(counts, *auc_pair_matrix(groups, "score"))
        for weights, value in zip(counts, calculated):
            explicit = pd.concat([g for g, weight in zip(groups, weights) for _ in range(weight)])
            self.assertAlmostEqual(value, roc_auc_score(explicit.target, explicit.score), places=14)

    def test_pairing_identical_models_gives_zero_difference(self):
        weeks = pd.date_range("2024-01-01", periods=8, freq="W-MON")
        counts, _ = draw_week_counts(weeks, 4, replicates=20)
        groups = [pd.DataFrame({"target": [0, 1], "score": [i / 10, .4]}) for i in range(8)]
        one = weighted_auc(counts, *auc_pair_matrix(groups, "score"))
        two = weighted_auc(counts, *auc_pair_matrix(groups, "score"))
        np.testing.assert_array_equal(one - two, np.zeros(20))

    def test_seed_reproducibility_and_invalid_blocks(self):
        weeks = pd.date_range("2024-01-01", periods=8, freq="W-MON")
        first, _ = draw_week_counts(weeks, 2, replicates=20)
        second, _ = draw_week_counts(weeks, 2, replicates=20)
        np.testing.assert_array_equal(first, second)
        with self.assertRaises(ValueError):
            calendar_blocks(weeks, 10)
        with self.assertRaises(ValueError):
            calendar_blocks(weeks[::-1], 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from thermonas.predictor import TimeStepPredictor


class PredictorTests(unittest.TestCase):
    def test_probabilities_are_finite_and_bounded(self) -> None:
        features = np.array(
            [
                [0.239, 0.082, 4_200_000.0, 1_129_600.0, 0.0001, 0.001],
                [1.0, 1.0, 3_000_000.0, 3_000_000.0, 0.0005, 0.004],
            ]
        )

        probabilities = TimeStepPredictor().predict_probabilities(features)

        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()

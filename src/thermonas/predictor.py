"""Lightweight NumPy inference for the bundled time-step classifier."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .solver import SimulationParameters

MODEL_PATH = Path(__file__).resolve().parent / "models" / "timestep_predictor.npz"


class TimeStepPredictor:
    """Run the exported dense neural network without TensorFlow."""

    def __init__(self) -> None:
        self._mean: NDArray[np.float64] | None = None
        self._scale: NDArray[np.float64] | None = None
        self._layers: tuple[tuple[NDArray[np.float32], NDArray[np.float32]], ...] = ()

    def _load(self) -> None:
        if self._mean is not None:
            return
        if not MODEL_PATH.is_file():
            raise RuntimeError("The packaged time-step predictor is missing.")

        with np.load(MODEL_PATH, allow_pickle=False) as model:
            self._mean = model["mean"].copy()
            self._scale = model["scale"].copy()
            self._layers = tuple(
                (model[f"weight_{index}"].copy(), model[f"bias_{index}"].copy())
                for index in range(4)
            )

        if self._mean.shape != (6,) or self._scale.shape != (6,):
            raise RuntimeError("The packaged time-step scaler has an invalid shape.")

    def predict_probabilities(
        self, features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        self._load()
        assert self._mean is not None and self._scale is not None

        values = (np.asarray(features, dtype=np.float64) - self._mean) / self._scale
        for weights, bias in self._layers[:-1]:
            values = np.maximum(values @ weights + bias, 0.0)
        output_weights, output_bias = self._layers[-1]
        logits = values @ output_weights + output_bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -80.0, 80.0)))

    def suggest(self, parameters: SimulationParameters) -> float:
        minimum, maximum, increment = 1e-5, 8e-3, 1e-5
        candidates = np.arange(minimum, maximum + increment / 2, increment)
        features = np.column_stack(
            (
                np.full_like(candidates, parameters.fluid_thermal_conductivity),
                np.full_like(candidates, parameters.solid_effective_thermal_conductivity),
                np.full_like(candidates, parameters.fluid_volumetric_heat_capacity),
                np.full_like(candidates, parameters.solid_volumetric_heat_capacity),
                np.full_like(candidates, parameters.velocity),
                candidates,
            )
        )
        unstable = np.flatnonzero(self.predict_probabilities(features) < 0.5)
        last_stable_index = max(0, int(unstable[0]) - 1) if unstable.size else len(candidates) - 1
        return float(f"{candidates[last_stable_index]:.2g}")

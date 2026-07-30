from __future__ import annotations

import unittest

import numpy as np

from thermonas.solver import SimulationParameters, solve


def parameters(**overrides: float | int) -> SimulationParameters:
    defaults = {
        "length": 0.02,
        "height": 0.01,
        "ambient_temperature_inlet": 20.0,
        "ambient_temperature_outlet": 20.0,
        "initial_fluid_temperature": 20.0,
        "initial_solid_temperature": 20.0,
        "fluid_thermal_conductivity": 0.239,
        "solid_effective_thermal_conductivity": 0.082,
        "solid_thermal_conductivity": 0.375,
        "volumetric_heat_transfer_coefficient": 0.0,
        "inlet_heat_transfer_coefficient": 0.0,
        "outlet_heat_transfer_coefficient": 0.0,
        "fluid_volumetric_heat_capacity": 4_200_000.0,
        "solid_volumetric_heat_capacity": 1_129_600.0,
        "porosity": 0.7,
        "velocity": 0.0001,
        "viscosity": 0.001003,
        "permeability": 8.435e-9,
        "time_steps": 4,
        "x_points": 5,
        "y_points": 4,
        "time_step": 1e-5,
        "lower_wall_temperature": 20.0,
        "upper_wall_temperature": 20.0,
    }
    defaults.update(overrides)
    return SimulationParameters(**defaults)


class SolverTests(unittest.TestCase):
    def test_uniform_equilibrium_is_preserved(self) -> None:
        result = solve(parameters())

        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.fluid_temperature, 20.0)
        np.testing.assert_allclose(result.solid_temperature, 20.0)

    def test_pressure_grid_supports_non_square_meshes(self) -> None:
        values = parameters(x_points=7, y_points=4)
        result = solve(values)

        self.assertEqual(result.pressure.shape, (7, 4))
        expected_drop = values.velocity * values.viscosity * values.length / values.permeability
        self.assertAlmostEqual(result.pressure[0, 0], expected_drop)
        self.assertAlmostEqual(result.pressure[-1, -1], 0.0)
        np.testing.assert_allclose(result.pressure[:, 0], result.pressure[:, -1])

    def test_invalid_porosity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Porosity"):
            solve(parameters(porosity=1.0))

    def test_grid_spacing_includes_both_channel_boundaries(self) -> None:
        values = parameters(length=0.03, height=0.02, x_points=7, y_points=5)

        self.assertAlmostEqual(values.dx, 0.005)
        self.assertAlmostEqual(values.dy, 0.005)


if __name__ == "__main__":
    unittest.main()

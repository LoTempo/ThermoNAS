"""Finite-difference solver for conjugate heat transfer in a porous channel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SimulationParameters:
    """Physical properties, boundary conditions, and grid settings."""

    length: float
    height: float
    ambient_temperature_inlet: float
    ambient_temperature_outlet: float
    initial_fluid_temperature: float
    initial_solid_temperature: float
    fluid_thermal_conductivity: float
    solid_effective_thermal_conductivity: float
    solid_thermal_conductivity: float
    volumetric_heat_transfer_coefficient: float
    inlet_heat_transfer_coefficient: float
    outlet_heat_transfer_coefficient: float
    fluid_volumetric_heat_capacity: float
    solid_volumetric_heat_capacity: float
    porosity: float
    velocity: float
    viscosity: float
    permeability: float
    time_steps: int
    x_points: int
    y_points: int
    time_step: float
    lower_wall_temperature: float
    upper_wall_temperature: float

    @property
    def dx(self) -> float:
        return self.length / (self.x_points - 1)

    @property
    def dy(self) -> float:
        return self.height / (self.y_points - 1)

    @property
    def estimated_temperature_storage_bytes(self) -> int:
        """Memory used by the two full temperature histories."""

        return 2 * self.time_steps * self.x_points * self.y_points * 8

    def validate(self) -> None:
        positive_values = {
            "length": self.length,
            "height": self.height,
            "fluid thermal conductivity": self.fluid_thermal_conductivity,
            "solid effective thermal conductivity": self.solid_effective_thermal_conductivity,
            "solid thermal conductivity": self.solid_thermal_conductivity,
            "fluid volumetric heat capacity": self.fluid_volumetric_heat_capacity,
            "solid volumetric heat capacity": self.solid_volumetric_heat_capacity,
            "viscosity": self.viscosity,
            "permeability": self.permeability,
            "time step": self.time_step,
        }
        for name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name.capitalize()} must be a positive finite value.")

        if not 0 < self.porosity < 1:
            raise ValueError("Porosity must be between 0 and 1.")
        if self.velocity < 0 or not np.isfinite(self.velocity):
            raise ValueError("Velocity must be a non-negative finite value.")
        if self.volumetric_heat_transfer_coefficient < 0:
            raise ValueError("The volumetric heat-transfer coefficient cannot be negative.")
        if self.inlet_heat_transfer_coefficient < 0 or self.outlet_heat_transfer_coefficient < 0:
            raise ValueError("External heat-transfer coefficients cannot be negative.")
        if self.time_steps < 2:
            raise ValueError("At least two time steps are required.")
        if self.x_points < 3 or self.y_points < 3:
            raise ValueError("At least three grid points are required in each direction.")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    fluid_temperature: FloatArray
    solid_temperature: FloatArray
    pressure: FloatArray
    converged: bool
    completed_steps: int


def solve(parameters: SimulationParameters) -> SimulationResult:
    """Solve the coupled fluid/solid energy equations with an explicit scheme."""

    parameters.validate()
    p = parameters

    fluid = np.full(
        (p.time_steps, p.x_points, p.y_points),
        p.initial_fluid_temperature,
        dtype=np.float64,
    )
    solid = np.full(
        (p.time_steps, p.x_points, p.y_points),
        p.initial_solid_temperature,
        dtype=np.float64,
    )
    fluid[:, 0, :] = p.ambient_temperature_inlet

    x = np.linspace(0.0, p.length, p.x_points)
    pressure_profile = (p.velocity * p.viscosity / p.permeability) * (p.length - x)
    pressure = np.repeat(pressure_profile[:, np.newaxis], p.y_points, axis=1)

    lower_bound = min(
        p.lower_wall_temperature,
        p.upper_wall_temperature,
        p.initial_fluid_temperature,
        p.initial_solid_temperature,
        p.ambient_temperature_inlet,
        p.ambient_temperature_outlet,
    )
    upper_bound = max(
        p.lower_wall_temperature,
        p.upper_wall_temperature,
        p.initial_fluid_temperature,
        p.initial_solid_temperature,
        p.ambient_temperature_inlet,
        p.ambient_temperature_outlet,
    )
    tolerance = 1e-9 * max(1.0, abs(lower_bound), abs(upper_bound))

    completed_steps = 1
    converged = True

    for step in range(1, p.time_steps):
        previous_solid = solid[step - 1]
        previous_fluid = fluid[step - 1]

        solid_laplacian = (
            (previous_solid[:-2, 1:-1] - 2 * previous_solid[1:-1, 1:-1] + previous_solid[2:, 1:-1])
            / p.dx**2
            + (
                previous_solid[1:-1, :-2]
                - 2 * previous_solid[1:-1, 1:-1]
                + previous_solid[1:-1, 2:]
            )
            / p.dy**2
        )
        fluid_laplacian = (
            (previous_fluid[:-2, 1:-1] - 2 * previous_fluid[1:-1, 1:-1] + previous_fluid[2:, 1:-1])
            / p.dx**2
            + (
                previous_fluid[1:-1, :-2]
                - 2 * previous_fluid[1:-1, 1:-1]
                + previous_fluid[1:-1, 2:]
            )
            / p.dy**2
        )

        solid[step, 1:-1, 1:-1] = previous_solid[1:-1, 1:-1] + p.time_step * (
            p.solid_effective_thermal_conductivity * solid_laplacian
            + p.volumetric_heat_transfer_coefficient
            * (previous_fluid[1:-1, 1:-1] - previous_solid[1:-1, 1:-1])
        ) / ((1 - p.porosity) * p.solid_volumetric_heat_capacity)

        fluid[step, 1:-1, 1:-1] = previous_fluid[1:-1, 1:-1] + p.time_step * (
            (
                p.fluid_thermal_conductivity * fluid_laplacian
                + p.volumetric_heat_transfer_coefficient
                * (previous_solid[1:-1, 1:-1] - previous_fluid[1:-1, 1:-1])
            )
            / (p.porosity * p.fluid_volumetric_heat_capacity)
            - p.velocity
            * (previous_fluid[1:-1, 1:-1] - previous_fluid[:-2, 1:-1])
            / p.dx
        )

        solid[step, 0, :] = (
            p.inlet_heat_transfer_coefficient * p.dx * p.ambient_temperature_inlet
            + p.solid_thermal_conductivity * solid[step, 1, :]
        ) / (p.solid_thermal_conductivity + p.inlet_heat_transfer_coefficient * p.dx)
        solid[step, -1, :] = (
            p.solid_thermal_conductivity * solid[step, -2, :]
            + p.outlet_heat_transfer_coefficient * p.dx * p.ambient_temperature_outlet
        ) / (p.solid_thermal_conductivity + p.outlet_heat_transfer_coefficient * p.dx)

        fluid[step, 0, :] = p.ambient_temperature_inlet
        fluid[step, -1, :] = fluid[step, -2, :]
        fluid[step, :, 0] = p.lower_wall_temperature
        fluid[step, :, -1] = p.upper_wall_temperature
        solid[step, :, 0] = p.lower_wall_temperature
        solid[step, :, -1] = p.upper_wall_temperature

        completed_steps = step + 1
        current_fluid = fluid[step]
        current_solid = solid[step]
        invalid = (
            not np.isfinite(current_fluid).all()
            or not np.isfinite(current_solid).all()
            or current_fluid.min() < lower_bound - tolerance
            or current_fluid.max() > upper_bound + tolerance
            or current_solid.min() < lower_bound - tolerance
            or current_solid.max() > upper_bound + tolerance
        )
        if invalid:
            converged = False
            break

    return SimulationResult(
        fluid_temperature=fluid[:completed_steps],
        solid_temperature=solid[:completed_steps],
        pressure=pressure,
        converged=converged,
        completed_steps=completed_steps,
    )

"""ThermoNAS: transient conjugate heat transfer in porous TPMS channels."""

from .solver import SimulationParameters, SimulationResult, solve

__all__ = ["SimulationParameters", "SimulationResult", "solve"]
__version__ = "1.0.0"

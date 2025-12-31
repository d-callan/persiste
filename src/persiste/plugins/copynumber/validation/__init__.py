"""Validation framework for CopyNumberDynamics plugin."""

from persiste.plugins.copynumber.validation.cn_simulator import (
    simulate_cn_evolution,
    SimulationScenario,
)

__all__ = [
    'simulate_cn_evolution',
    'SimulationScenario',
]

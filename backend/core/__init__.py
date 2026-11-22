"""Núcleo numérico del simulador de trayectorias con splines.

Todas las funciones son puras (sin I/O ni estado global), vectorizadas con NumPy
y no dependen de FastAPI. Las usan tanto la API como el notebook del informe.

Prohibido usar ``scipy.interpolate`` u otra librería de interpolación aquí:
los splines se implementan a mano.
"""

from . import (
    arc_length,
    convergence,
    cubic_natural,
    hermite,
    kinematics,
    parametrization,
    tangents,
    trajectory,
    velocity_profile,
)

__all__ = [
    "arc_length",
    "convergence",
    "cubic_natural",
    "hermite",
    "kinematics",
    "parametrization",
    "tangents",
    "trajectory",
    "velocity_profile",
]

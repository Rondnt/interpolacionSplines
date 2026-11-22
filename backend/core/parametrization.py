"""Parametrización de la trayectoria: asignación de los nodos ``t_k`` a cada waypoint.

Cada coordenada de la trayectoria se interpola por separado contra un parámetro
creciente ``t``. La elección de ``t_k`` cambia la forma de la curva:

* Uniforme   (α = 0):   t_{k+1} = t_k + 1
* Cuerda     (α = 1):   t_{k+1} = t_k + ‖P_{k+1} − P_k‖
* Centrípeta (α = 1/2): t_{k+1} = t_k + ‖P_{k+1} − P_k‖^{1/2}

Las tres son casos de la familia general t_{k+1} = t_k + ‖P_{k+1} − P_k‖^α con t_0 = 0.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Parametrization = Literal["uniform", "chord", "centripetal"]

ALPHA: dict[str, float] = {"uniform": 0.0, "chord": 1.0, "centripetal": 0.5}


def parametrize(P: np.ndarray, mode: Parametrization = "centripetal") -> np.ndarray:
    """Calcula los nodos ``t_nodes`` para los waypoints ``P``.

    Fórmula implementada (familia α)::

        t_0 = 0,    t_{k+1} = t_k + ‖P_{k+1} − P_k‖^α,
        α = 0 (uniforme), 1 (cuerda), 1/2 (centrípeta).

    Con α = 0 se obtiene t_k = k. La parametrización centrípeta reparte mejor el
    parámetro cuando las distancias entre waypoints son muy distintas y, combinada
    con las tangentes de Catmull-Rom no uniformes, evita bucles y cúspides
    (Yuksel, Schaefer y Keyser, 2011).

    Parámetros
    ----------
    P : np.ndarray, forma (n+1, d)
        Waypoints, una fila por punto.
    mode : {"uniform", "chord", "centripetal"}
        Tipo de parametrización.

    Retorna
    -------
    t_nodes : np.ndarray, forma (n+1,)
        Nodos estrictamente crecientes.

    Lanza
    -----
    ValueError
        Si hay menos de 2 waypoints, el modo no existe o dos waypoints
        consecutivos coinciden (Δ_k = 0 en cuerda/centrípeta).

    Complejidad: O(n).
    """
    if mode not in ALPHA:
        raise ValueError(f"Parametrización desconocida: {mode!r}. Use uniform, chord o centripetal.")
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    if P.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 waypoints.")

    dist = np.linalg.norm(np.diff(P, axis=0), axis=1)
    if mode != "uniform" and np.any(dist <= 0.0):
        k = int(np.argmax(dist <= 0.0))
        raise ValueError(f"Los waypoints {k} y {k + 1} coinciden; la parametrización '{mode}' requiere puntos distintos.")

    delta = dist ** ALPHA[mode]
    t_nodes = np.concatenate(([0.0], np.cumsum(delta)))
    return t_nodes

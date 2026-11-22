"""Orquestación: de una lista de waypoints a la trayectoria completa con su análisis.

Encadena los módulos del núcleo en el mismo orden que el modelo matemático:

1. Parametrización:   t_nodes = parametrize(P)
2. Spline:            Hermite (tangentes cardinales) o cúbico natural/sujeto
3. Muestreo:          N puntos por tramo, u ∈ [0, 1), más el último nodo
4. Cinemática:        ‖p'‖, ‖p''‖ y κ; los picos de κ entre muestras se refinan (sección áurea)
5. Longitud de arco:  s(t) con Simpson compuesto
6. Velocidad:         v_max por curvatura (incluyendo los picos) y perfil con límite longitudinal
7. Tiempo:            t(s) acumulado y tiempo total

La usan la API (``api/main.py``) y el notebook del informe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from . import arc_length, cubic_natural, hermite, kinematics, tangents, velocity_profile
from .parametrization import Parametrization, parametrize

Method = Literal["hermite", "natural", "clamped"]
EvalFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Spline:
    """Spline ya construido: nodos, waypoints, tangentes en los nodos y evaluador."""

    t_nodes: np.ndarray
    P: np.ndarray
    tangents: np.ndarray
    evaluate: EvalFn

    def speed(self, t: np.ndarray) -> np.ndarray:
        """Rapidez paramétrica ‖p'(t)‖ (para integrar la longitud de arco)."""
        return kinematics.speed(self.evaluate(np.asarray(t, dtype=float))[1])

    def curvature(self, t: np.ndarray) -> np.ndarray:
        """Curvatura κ(t) (para refinar sus picos)."""
        _, d1, d2 = self.evaluate(np.asarray(t, dtype=float))
        return kinematics.curvature(d1, d2)


@dataclass(frozen=True)
class TrajectoryResult:
    """Resultado completo de la trayectoria (todos los arreglos tienen N filas)."""

    points: np.ndarray        # (N, 2) posición
    t: np.ndarray             # (N,)  parámetro
    s: np.ndarray             # (N,)  longitud de arco acumulada [m]
    speed: np.ndarray         # (N,)  ‖p'(t)‖ (paramétrica)
    acceleration: np.ndarray  # (N,)  ‖p''(t)‖ (paramétrica)
    curvature: np.ndarray     # (N,)  κ [1/m] en las muestras
    v_max: np.ndarray         # (N,)  límite por curvatura [m/s] (considera los picos entre muestras)
    v_profile: np.ndarray     # (N,)  perfil factible [m/s]
    time: np.ndarray          # (N,)  tiempo acumulado [s]
    tangents: np.ndarray      # (n+1, 2) tangentes en los nodos
    knots_index: np.ndarray   # (n+1,) índices de ``points`` que son waypoints
    t_nodes: np.ndarray       # (n+1,) nodos
    length: float
    time_estimate: float
    max_curvature: float


def build_spline(
    P: np.ndarray,
    method: Method = "hermite",
    tension: float = 0.0,
    parametrization: Parametrization = "centripetal",
    end_tangents: np.ndarray | None = None,
) -> Spline:
    """Construye el spline que interpola los waypoints ``P``.

    * ``hermite``: tangentes cardinales (Catmull-Rom no uniforme si c = 0).
    * ``natural``: spline cúbico con M_0 = M_n = 0.
    * ``clamped``: spline cúbico con pendientes en los extremos. Si no se dan
      ``end_tangents``, se usan las diferencias unilaterales
      m_0 = (P_1 − P_0)/Δ_0 y m_n = (P_n − P_{n−1})/Δ_{n−1}.

    Parámetros
    ----------
    P : np.ndarray, forma (n+1, 2)
    method : {"hermite", "natural", "clamped"}
    tension : float
        Tensión c ∈ [0, 1] (solo Hermite).
    parametrization : {"uniform", "chord", "centripetal"}
    end_tangents : np.ndarray de forma (2, 2), opcional
        [m_0, m_n] para el spline sujeto, respecto del parámetro t.

    Retorna
    -------
    Spline

    Complejidad: O(n).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("Los waypoints deben tener forma (n+1, 2).")
    t_nodes = parametrize(P, parametrization)

    if method == "hermite":
        m = tangents.cardinal_tangents(t_nodes, P, tension)
        return Spline(t_nodes, P, m, lambda t: hermite.evaluate(t_nodes, P, m, t))

    if method == "natural":
        M = cubic_natural.natural_second_derivatives(t_nodes, P)
    elif method == "clamped":
        if end_tangents is None:
            delta = np.diff(t_nodes)
            m0 = (P[1] - P[0]) / delta[0]
            mn = (P[-1] - P[-2]) / delta[-1]
        else:
            m0, mn = np.asarray(end_tangents, dtype=float)
        M = cubic_natural.clamped_second_derivatives(t_nodes, P, m0, mn)
    else:
        raise ValueError(f"Método desconocido: {method!r}. Use hermite, natural o clamped.")

    m = cubic_natural.nodal_slopes(t_nodes, P, M)
    return Spline(t_nodes, P, m, lambda t: cubic_natural.evaluate(t_nodes, P, M, t))


def sample_parameters(t_nodes: np.ndarray, samples_per_segment: int) -> tuple[np.ndarray, np.ndarray]:
    """Malla de parámetros: ``samples_per_segment`` puntos por tramo más el último nodo.

    En cada tramo k se toma u_j = j/N, j = 0..N−1, y t = t_k + u_j·Δ_k. Al final
    se agrega t_n. Total: n·N + 1 puntos; el waypoint k está en el índice k·N.

    Retorna
    -------
    (t, knots_index) : np.ndarray de forma (n·N+1,) y (n+1,).

    Complejidad: O(n·N).
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    N = int(samples_per_segment)
    u = np.arange(N) / N
    delta = np.diff(t_nodes)
    t = (t_nodes[:-1, None] + u[None, :] * delta[:, None]).ravel()
    t = np.append(t, t_nodes[-1])
    knots_index = np.arange(t_nodes.shape[0]) * N
    return t, knots_index


def compute_trajectory(
    waypoints: np.ndarray,
    method: Method = "hermite",
    tension: float = 0.0,
    parametrization: Parametrization = "centripetal",
    samples_per_segment: int = 50,
    v_cruise: float = 5.0,
    a_lat_max: float = 2.0,
    a_long_max: float = 1.5,
    end_tangents: np.ndarray | None = None,
) -> TrajectoryResult:
    """Calcula la trayectoria completa y su análisis cinemático.

    Ver el docstring del módulo para la secuencia de pasos. El perfil de
    velocidad parte y termina en reposo.

    Parámetros
    ----------
    waypoints : np.ndarray, forma (n+1, 2)
    method, tension, parametrization, end_tangents
        Ver ``build_spline``.
    samples_per_segment : int
        Puntos por tramo (≥ 2).
    v_cruise, a_lat_max, a_long_max : float
        Límites físicos (> 0) en m/s y m/s².

    Retorna
    -------
    TrajectoryResult (todos los valores son finitos).

    Complejidad: O(n·N).
    """
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment debe ser ≥ 2.")
    if min(v_cruise, a_lat_max, a_long_max) <= 0:
        raise ValueError("Los límites físicos deben ser positivos.")

    spline = build_spline(waypoints, method, tension, parametrization, end_tangents)
    t, knots_index = sample_parameters(spline.t_nodes, samples_per_segment)
    points, d1, d2 = spline.evaluate(t)

    speed = kinematics.speed(d1)
    accel = kinematics.acceleration_norm(d2)
    kappa = kinematics.curvature(d1, d2)

    # Picos de κ entre muestras: se ubican con sección áurea y se imponen en las dos
    # muestras que los rodean. Así v (interpolada linealmente entre muestras) respeta
    # a_lat también en el pico, aunque este sea más angosto que la malla.
    t_peak, kappa_peak = kinematics.curvature_peaks(spline.curvature, t, kappa)
    j = np.clip(np.searchsorted(t, t_peak, side="right") - 1, 0, t.shape[0] - 2)
    kappa_env = kappa.copy()
    np.maximum.at(kappa_env, j, kappa_peak)
    np.maximum.at(kappa_env, j + 1, kappa_peak)

    s = arc_length.cumulative_arc_length(t, spline.speed)
    v_max = velocity_profile.curvature_speed_limit(kappa_env, v_cruise, a_lat_max)
    v = velocity_profile.longitudinal_limit(v_max, s, a_long_max)
    time = velocity_profile.cumulative_time(s, v, a_long_max)

    return TrajectoryResult(
        points=points,
        t=t,
        s=s,
        speed=speed,
        acceleration=accel,
        curvature=kappa,
        v_max=v_max,
        v_profile=v,
        time=time,
        tangents=spline.tangents,
        knots_index=knots_index,
        t_nodes=spline.t_nodes,
        length=float(s[-1]),
        time_estimate=float(time[-1]),
        max_curvature=float(kappa_env.max()),
    )

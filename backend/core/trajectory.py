"""Orquestación: de una lista de waypoints a la trayectoria completa con su análisis.

Encadena los módulos del núcleo en el mismo orden que el modelo matemático:

1. Parametrización:   t_nodes = parametrize(P)
2. Spline:            Hermite (tangentes cardinales) o cúbico natural/sujeto
3. Muestreo:          N puntos por tramo, u ∈ [0, 1), más el último nodo
4. Cinemática:        ‖p'‖, ‖p''‖ y κ; máximo de κ en cada intervalo entre muestras (sección áurea)
5. Longitud de arco:  s(t) con Simpson compuesto
6. Velocidad:         v_max por curvatura y perfil longitudinal en una malla densificada en los picos
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
    v_max: np.ndarray         # (N,)  límite por curvatura [m/s]
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


@dataclass(frozen=True)
class Profile:
    """Perfil de velocidad sobre la malla interna (contiene todas las muestras)."""

    t: np.ndarray
    s: np.ndarray
    v_max: np.ndarray
    v: np.ndarray
    time: np.ndarray
    max_curvature: float


def _interval_envelope(spline: Spline, t: np.ndarray, iters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (κ̂_j, t̂_j, κ_env) con κ_env,i = máx(κ̂_{i−1}, κ̂_i). Complejidad: O(N·iters)."""
    kappa_int, t_int = kinematics.interval_max_curvature(spline.curvature, t, spline.curvature(t), iters)
    env = np.maximum(np.concatenate((kappa_int[:1], kappa_int)), np.concatenate((kappa_int, kappa_int[-1:])))
    return kappa_int, t_int, env


def feasible_profile(
    spline: Spline,
    t: np.ndarray,
    v_cruise: float,
    a_lat_max: float,
    a_long_max: float,
    levels: int = 12,
) -> Profile:
    """Perfil de velocidad calculado en una malla interna densificada en los picos de κ.

    1. Se ubican los máximos locales de κ por intervalo y se agregan, alrededor de cada
       pico y de cada nodo t*, los puntos graduados::

           t* ± h·2^{−j},   j = 0..levels,   h = separación local entre muestras

    2. En cada intervalo de esa malla se usa κ̂_j = máx κ, y en cada punto
       κ_env,i = máx(κ̂_{i−1}, κ̂_i). Si v_i y v_{i+1} cumplen v ≤ √(a_lat/κ̂_i), toda
       interpolación lineal de v (o de v²) en el intervalo también lo cumple.

    Sin la densificación, el límite de un pico muy angosto (por ejemplo las esquinas
    con tensión c = 1) se aplicaría a dos intervalos completos de muestreo y el tiempo
    quedaría muy sobrestimado. Con ella, solo se aplica a un entorno de ancho h·2^{−levels}.

    Parámetros
    ----------
    spline : Spline
    t : np.ndarray, forma (N,)
        Parámetros de las muestras.
    v_cruise, a_lat_max, a_long_max : float
        Límites físicos.
    levels : int
        Niveles de densificación.

    Retorna
    -------
    Profile

    Complejidad: O((N + K·levels)·iters), con K = número de picos y nodos.
    """
    kappa_int, t_int, _ = _interval_envelope(spline, t, iters=20)
    prev = np.concatenate(([-np.inf], kappa_int[:-1]))
    nxt = np.concatenate((kappa_int[1:], [-np.inf]))
    is_peak = (kappa_int >= prev) & (kappa_int >= nxt) & (kappa_int > 0)
    centers = np.concatenate((spline.t_nodes, t_int[is_peak]))
    j = np.clip(np.searchsorted(t, centers, side="right") - 1, 0, t.shape[0] - 2)
    offsets = (t[j + 1] - t[j])[:, None] * 0.5 ** np.arange(levels + 1)[None, :]
    extra = np.concatenate((centers, (centers[:, None] - offsets).ravel(), (centers[:, None] + offsets).ravel()))
    grid = np.union1d(t, extra[(extra > t[0]) & (extra < t[-1])])

    _, _, env = _interval_envelope(spline, grid, iters=12)
    s = arc_length.cumulative_arc_length(grid, spline.speed, n_sub=8)
    v_max = velocity_profile.curvature_speed_limit(env, v_cruise, a_lat_max)
    v = velocity_profile.longitudinal_limit(v_max, s, a_long_max)
    time = velocity_profile.cumulative_time(s, v, a_long_max)
    return Profile(grid, s, v_max, v, time, float(env.max()))


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

    prof = feasible_profile(spline, t, v_cruise, a_lat_max, a_long_max)
    idx = np.searchsorted(prof.t, t)      # las muestras pertenecen a la malla interna
    s, v_max, v, time = prof.s[idx], prof.v_max[idx], prof.v[idx], prof.time[idx]

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
        max_curvature=prof.max_curvature,
    )

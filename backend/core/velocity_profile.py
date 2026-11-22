"""Perfil de velocidad físicamente factible para recorrer la trayectoria.

Dos restricciones físicas:

1. Lateral (no derrapar o volcar en las curvas): a_lat = v²·κ ≤ a_lat_max
   ⇒ v ≤ √(a_lat_max/κ).
2. Longitudinal (no acelerar ni frenar más de lo posible): |dv/dt| ≤ a_long_max
   ⇒ v_i² ≤ v_{i−1}² + 2·a_long_max·Δs_i (ecuación de Torricelli).

El dron parte del reposo y termina en reposo.
"""

from __future__ import annotations

import numpy as np


def curvature_speed_limit(kappa: np.ndarray, v_cruise: float, a_lat_max: float) -> np.ndarray:
    """Límite de velocidad impuesto por la curvatura.

    Fórmula::

        v_max(s) = min( v_crucero, √(a_lat_max / κ(s)) )

    En rectas (κ = 0) el límite es v_crucero. Con κ muy grande (esquinas) el
    límite tiende a 0.

    Parámetros
    ----------
    kappa : np.ndarray, forma (N,)
        Curvatura (≥ 0, finita).
    v_cruise : float
        Velocidad de crucero (> 0), en m/s.
    a_lat_max : float
        Aceleración lateral máxima (> 0), en m/s².

    Retorna
    -------
    v_max : np.ndarray, forma (N,)

    Complejidad: O(N).
    """
    kappa = np.asarray(kappa, dtype=float)
    with np.errstate(divide="ignore"):
        v_lat = np.where(kappa > 0, np.sqrt(a_lat_max / np.where(kappa > 0, kappa, 1.0)), np.inf)
    return np.minimum(v_cruise, v_lat)


def longitudinal_limit(
    v_max: np.ndarray,
    s: np.ndarray,
    a_long_max: float,
    v_start: float = 0.0,
    v_end: float = 0.0,
) -> np.ndarray:
    """Aplica el límite de aceleración y frenado con pasadas hacia adelante y hacia atrás.

    Pasada hacia adelante (acelerar), con w_i = v_i²::

        w_0 = min(v_max_0², v_start²),   w_i = min( v_max_i², w_{i−1} + 2·a·Δs_i )

    Pasada hacia atrás (frenar a tiempo)::

        w_N = min(w_N, v_end²),          w_i = min( w_i, w_{i+1} + 2·a·Δs_{i+1} )

    Vectorización: con c_i = 2·a·s_i, la recurrencia hacia adelante equivale a::

        w_i − c_i = min( v_max_i² − c_i, w_{i−1} − c_{i−1} )
        ⇒ w_i = c_i + mín_{j ≤ i}(g_j),   g_j = v_max_j² − c_j  (g_0 con w_0)

    que se calcula con un mínimo acumulado (``np.minimum.accumulate``), sin bucles.
    La pasada hacia atrás es análoga con c'_i = 2·a·(L − s_i) y el arreglo invertido.

    Parámetros
    ----------
    v_max : np.ndarray, forma (N,)
        Límite de velocidad en cada punto (por ejemplo, el de la curvatura).
    s : np.ndarray, forma (N,)
        Longitud de arco acumulada (creciente).
    a_long_max : float
        Aceleración longitudinal máxima (> 0).
    v_start, v_end : float
        Velocidad inicial y final (0 = parte y llega en reposo).

    Retorna
    -------
    v : np.ndarray, forma (N,)
        Perfil factible: v ≤ v_max y |Δ(v²)| ≤ 2·a·Δs entre puntos vecinos.

    Complejidad: O(N).
    """
    v_max = np.asarray(v_max, dtype=float)
    s = np.asarray(s, dtype=float)
    w = v_max**2

    # Hacia adelante
    c = 2.0 * a_long_max * s
    g = w - c
    g[0] = min(w[0], v_start**2) - c[0]
    w = c + np.minimum.accumulate(g)

    # Hacia atrás
    c = 2.0 * a_long_max * (s[-1] - s)
    g = (w - c)[::-1]
    g[0] = min(w[-1], v_end**2) - c[-1]
    w = c + np.minimum.accumulate(g)[::-1]

    return np.sqrt(np.maximum(w, 0.0))


def cumulative_time(s: np.ndarray, v: np.ndarray, a_long_max: float) -> np.ndarray:
    """Tiempo acumulado para recorrer la trayectoria con el perfil v(s).

    Entre muestras consecutivas se supone aceleración constante, para la cual el
    tiempo exacto es la distancia sobre la velocidad media::

        Δt_i = Δs_i / ((v_{i−1} + v_i)/2),       T = Σ Δt_i

    (la fórmula Δs_i/v_i del plan original diverge cuando el dron parte del reposo).
    Si v_{i−1} = v_i = 0 (dos paradas seguidas) el tramo se hace acelerando y
    frenando a a_long_max: Δt_i = 2·√(Δs_i/a_long_max).

    Parámetros
    ----------
    s : np.ndarray, forma (N,)
        Longitud de arco acumulada.
    v : np.ndarray, forma (N,)
        Perfil de velocidad.
    a_long_max : float
        Aceleración longitudinal máxima (> 0).

    Retorna
    -------
    time : np.ndarray, forma (N,)
        time[0] = 0 y time[-1] = T, tiempo total estimado.

    Complejidad: O(N).
    """
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    ds = np.diff(s)
    v_sum = v[:-1] + v[1:]
    stopped = v_sum <= 1e-12
    dt = np.where(stopped, 2.0 * np.sqrt(ds / a_long_max), 2.0 * ds / np.where(stopped, 1.0, v_sum))
    return np.concatenate(([0.0], np.cumsum(dt)))

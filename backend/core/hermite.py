"""Spline de Hermite cúbico por tramos (continuidad C¹).

En el tramo [t_k, t_{k+1}], con Δ_k = t_{k+1} − t_k y u = (t − t_k)/Δ_k ∈ [0, 1]::

    p(u) = h00(u)·P_k + h10(u)·Δ_k·m_k + h01(u)·P_{k+1} + h11(u)·Δ_k·m_{k+1}

donde P_k son los waypoints y m_k las tangentes (derivadas respecto de t).
La curva pasa por todos los P_k y su primera derivada vale m_k en cada nodo,
por lo que es C¹; la segunda derivada en general salta en los nodos.
"""

from __future__ import annotations

import numpy as np


def hermite_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bases cúbicas de Hermite.

    Fórmulas::

        h00 = 2u³ − 3u² + 1      h10 = u³ − 2u² + u
        h01 = −2u³ + 3u²         h11 = u³ − u²

    Cumplen h00(0) = h01(1) = 1, h00(1) = h01(0) = 0 y h10, h11 se anulan en
    u = 0 y u = 1 (solo aportan pendiente).

    Parámetros
    ----------
    u : np.ndarray
        Parámetro local en [0, 1].

    Retorna
    -------
    (h00, h10, h01, h11) : tupla de np.ndarray con la forma de ``u``.

    Complejidad: O(N).
    """
    u = np.asarray(u, dtype=float)
    u2 = u * u
    u3 = u2 * u
    return 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2


def hermite_basis_d1(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Primeras derivadas de las bases de Hermite respecto de u.

    Fórmulas::

        h00' = 6u² − 6u      h10' = 3u² − 4u + 1
        h01' = −6u² + 6u     h11' = 3u² − 2u

    Complejidad: O(N).
    """
    u = np.asarray(u, dtype=float)
    u2 = u * u
    return 6 * u2 - 6 * u, 3 * u2 - 4 * u + 1, -6 * u2 + 6 * u, 3 * u2 - 2 * u


def hermite_basis_d2(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Segundas derivadas de las bases de Hermite respecto de u.

    Fórmulas::

        h00'' = 12u − 6      h10'' = 6u − 4
        h01'' = −12u + 6     h11'' = 6u − 2

    Complejidad: O(N).
    """
    u = np.asarray(u, dtype=float)
    return 12 * u - 6, 6 * u - 4, -12 * u + 6, 6 * u - 2


def locate_segments(t_nodes: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Índice del tramo k tal que t_k ≤ t < t_{k+1} para cada valor de ``t_eval``.

    Convención en los nodos: t = t_k pertenece al tramo k (límite por la derecha),
    salvo el último nodo t_n, que pertenece al tramo n−1. Los valores fuera de
    [t_0, t_n] se asignan al primer o último tramo (extrapolación cúbica).

    Implementación: búsqueda binaria vectorizada (``np.searchsorted``).

    Complejidad: O(N log n).
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    k = np.searchsorted(t_nodes, np.asarray(t_eval, dtype=float), side="right") - 1
    return np.clip(k, 0, t_nodes.shape[0] - 2)


def evaluate(
    t_nodes: np.ndarray,
    P: np.ndarray,
    m: np.ndarray,
    t_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evalúa el spline de Hermite y sus dos primeras derivadas respecto de t.

    Fórmulas, con k el tramo de cada t, Δ_k = t_{k+1} − t_k y u = (t − t_k)/Δ_k::

        p(t)   = h00·P_k + h10·Δ_k·m_k + h01·P_{k+1} + h11·Δ_k·m_{k+1}
        p'(t)  = (1/Δ_k)  · [h00'·P_k + h10'·Δ_k·m_k + h01'·P_{k+1} + h11'·Δ_k·m_{k+1}]
        p''(t) = (1/Δ_k²) · [h00''·P_k + h10''·Δ_k·m_k + h01''·P_{k+1} + h11''·Δ_k·m_{k+1}]

    (regla de la cadena: du/dt = 1/Δ_k).

    Parámetros
    ----------
    t_nodes : np.ndarray, forma (n+1,)
        Nodos estrictamente crecientes.
    P : np.ndarray, forma (n+1, d) o (n+1,)
        Valores en los nodos (waypoints).
    m : np.ndarray, misma forma que ``P``
        Tangentes dp/dt en los nodos.
    t_eval : np.ndarray, forma (N,)
        Parámetros donde evaluar.

    Retorna
    -------
    (pos, d1, d2) : np.ndarray de forma (N, d), o (N,) si ``P`` es 1-D.
        Posición, primera y segunda derivada respecto de t. En un nodo interior,
        ``d2`` corresponde al límite por la derecha (ver ``locate_segments``).

    Complejidad: O(N log n) por la búsqueda del tramo; O(N) el resto.
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    P_in = np.asarray(P, dtype=float)
    one_dim = P_in.ndim == 1
    P2 = P_in[:, None] if one_dim else P_in
    m2 = np.asarray(m, dtype=float).reshape(P2.shape)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))

    k = locate_segments(t_nodes, t_eval)
    delta = (t_nodes[k + 1] - t_nodes[k])[:, None]
    u = (t_eval - t_nodes[k])[:, None] / delta

    Pk, Pk1 = P2[k], P2[k + 1]
    Tk, Tk1 = delta * m2[k], delta * m2[k + 1]   # tangentes escaladas Δ_k·m_k

    h00, h10, h01, h11 = hermite_basis(u)
    pos = h00 * Pk + h10 * Tk + h01 * Pk1 + h11 * Tk1

    h00, h10, h01, h11 = hermite_basis_d1(u)
    d1 = (h00 * Pk + h10 * Tk + h01 * Pk1 + h11 * Tk1) / delta

    h00, h10, h01, h11 = hermite_basis_d2(u)
    d2 = (h00 * Pk + h10 * Tk + h01 * Pk1 + h11 * Tk1) / delta**2

    if one_dim:
        return pos[:, 0], d1[:, 0], d2[:, 0]
    return pos, d1, d2


def second_derivative_jumps(t_nodes: np.ndarray, P: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Salto de la segunda derivada en cada nodo interior.

    Fórmula, para k = 1..n−1::

        salto_k = p''(t_k⁺) − p''(t_k⁻)
        p''(t_k⁺) = [−6·P_k − 4Δ_k·m_k + 6·P_{k+1} − 2Δ_k·m_{k+1}] / Δ_k²            (u = 0 del tramo k)
        p''(t_k⁻) = [ 6·P_{k−1} + 2Δ_{k−1}·m_{k−1} − 6·P_k + 4Δ_{k−1}·m_k] / Δ_{k−1}²  (u = 1 del tramo k−1)

    Un spline C² tendría todos los saltos nulos; Hermite con tangentes
    estimadas en general no.

    Retorna
    -------
    np.ndarray, forma (n−1, d)

    Complejidad: O(n).
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    P2 = np.asarray(P, dtype=float)
    P2 = P2[:, None] if P2.ndim == 1 else P2
    m2 = np.asarray(m, dtype=float).reshape(P2.shape)
    delta = np.diff(t_nodes)[:, None]

    right = (-6 * P2[:-1] - 4 * delta * m2[:-1] + 6 * P2[1:] - 2 * delta * m2[1:]) / delta**2
    left = (6 * P2[:-1] + 2 * delta * m2[:-1] - 6 * P2[1:] + 4 * delta * m2[1:]) / delta**2
    # right[k] = p''(t_k⁺) en el tramo k; left[k] = p''(t_{k+1}⁻) en el tramo k.
    return right[1:] - left[:-1]

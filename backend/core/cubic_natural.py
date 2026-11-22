"""Spline cúbico de interpolación (continuidad C²): natural y sujeto (clamped).

Se calculan las segundas derivadas ``M_k = S''(t_k)`` en los nodos resolviendo un
sistema tridiagonal con el algoritmo de Thomas, implementado a mano.

En el tramo [t_k, t_{k+1}], con Δ_k = t_{k+1} − t_k::

    S(t) = M_k·(t_{k+1} − t)³/(6Δ_k) + M_{k+1}·(t − t_k)³/(6Δ_k)
         + (y_k/Δ_k − M_k·Δ_k/6)·(t_{k+1} − t) + (y_{k+1}/Δ_k − M_{k+1}·Δ_k/6)·(t − t_k)

Condiciones en los nodos interiores (continuidad de S'), k = 1..n−1::

    Δ_{k−1}·M_{k−1} + 2(Δ_{k−1} + Δ_k)·M_k + Δ_k·M_{k+1}
        = 6·[ (y_{k+1} − y_k)/Δ_k − (y_k − y_{k−1})/Δ_{k−1} ]

Condiciones de frontera:
* Natural: M_0 = M_n = 0 (curvatura nula en los extremos).
* Sujeto:  S'(t_0) = m_0 y S'(t_n) = m_n dadas.
"""

from __future__ import annotations

import numpy as np

from .hermite import locate_segments


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Resuelve un sistema tridiagonal A·x = d con el algoritmo de Thomas.

    El sistema de N ecuaciones es::

        a_i·x_{i−1} + b_i·x_i + c_i·x_{i+1} = d_i,     i = 0..N−1,   (a_0 = c_{N−1} = 0)

    Eliminación hacia adelante (i = 1..N−1)::

        w    = a_i / b'_{i−1}
        b'_i = b_i − w·c_{i−1}
        d'_i = d_i − w·d'_{i−1}

    Sustitución hacia atrás::

        x_{N−1} = d'_{N−1} / b'_{N−1}
        x_i     = (d'_i − c_i·x_{i+1}) / b'_i

    Es estable sin pivoteo si A es estrictamente diagonal dominante, lo que se
    cumple en los sistemas del spline: |2(Δ_{k−1}+Δ_k)| > |Δ_{k−1}| + |Δ_k|.

    Parámetros
    ----------
    a : np.ndarray, forma (N,)
        Subdiagonal (a[0] se ignora).
    b : np.ndarray, forma (N,)
        Diagonal principal.
    c : np.ndarray, forma (N,)
        Superdiagonal (c[N−1] se ignora).
    d : np.ndarray, forma (N,) o (N, k)
        Lado derecho; con varias columnas se resuelven k sistemas a la vez
        (por ejemplo, las coordenadas x e y).

    Retorna
    -------
    x : np.ndarray con la forma de ``d``.

    Complejidad: O(N) (frente a O(N³) de la eliminación gaussiana densa).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    N = b.shape[0]
    if not (a.shape[0] == c.shape[0] == d.shape[0] == N):
        raise ValueError("a, b, c y d deben tener la misma longitud.")

    bp = b.copy()
    dp = d.copy()
    for i in range(1, N):
        w = a[i] / bp[i - 1]
        bp[i] = b[i] - w * c[i - 1]
        dp[i] = d[i] - w * dp[i - 1]

    x = np.empty_like(dp)
    x[-1] = dp[-1] / bp[-1]
    for i in range(N - 2, -1, -1):
        x[i] = (dp[i] - c[i] * x[i + 1]) / bp[i]
    return x


def _prepare(t_nodes: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    t_nodes = np.asarray(t_nodes, dtype=float)
    P = np.asarray(P, dtype=float)
    one_dim = P.ndim == 1
    P2 = P[:, None] if one_dim else P
    if P2.shape[0] != t_nodes.shape[0] or P2.shape[0] < 2:
        raise ValueError("t_nodes y P deben tener la misma cantidad (≥ 2) de nodos.")
    delta = np.diff(t_nodes)
    if np.any(delta <= 0):
        raise ValueError("t_nodes debe ser estrictamente creciente.")
    return t_nodes, P2, delta, one_dim


def natural_second_derivatives(t_nodes: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Segundas derivadas ``M`` del spline cúbico natural.

    Sistema para las incógnitas interiores M_1..M_{n−1} (con M_0 = M_n = 0)::

        Δ_{k−1}·M_{k−1} + 2(Δ_{k−1} + Δ_k)·M_k + Δ_k·M_{k+1}
            = 6·[ (y_{k+1} − y_k)/Δ_k − (y_k − y_{k−1})/Δ_{k−1} ]

    Parámetros
    ----------
    t_nodes : np.ndarray, forma (n+1,)
    P : np.ndarray, forma (n+1, d) o (n+1,)

    Retorna
    -------
    M : np.ndarray, misma forma que ``P``.

    Complejidad: O(n).
    """
    t_nodes, P2, delta, one_dim = _prepare(t_nodes, P)
    n = P2.shape[0] - 1
    M = np.zeros_like(P2)
    if n >= 2:
        slope = np.diff(P2, axis=0) / delta[:, None]          # (n, d)
        rhs = 6.0 * (slope[1:] - slope[:-1])                   # (n−1, d)
        a = np.concatenate(([0.0], delta[1:-1]))
        b = 2.0 * (delta[:-1] + delta[1:])
        c = np.concatenate((delta[1:-1], [0.0]))
        M[1:-1] = thomas_solve(a, b, c, rhs)
    return M[:, 0] if one_dim else M


def clamped_second_derivatives(
    t_nodes: np.ndarray,
    P: np.ndarray,
    m_start: np.ndarray,
    m_end: np.ndarray,
) -> np.ndarray:
    """Segundas derivadas ``M`` del spline cúbico sujeto (pendientes dadas en los extremos).

    Además de las ecuaciones interiores del spline natural, las fronteras son::

        2Δ_0·M_0 + Δ_0·M_1                 = 6·[ (y_1 − y_0)/Δ_0 − m_0 ]
        Δ_{n−1}·M_{n−1} + 2Δ_{n−1}·M_n     = 6·[ m_n − (y_n − y_{n−1})/Δ_{n−1} ]

    Se obtiene un sistema tridiagonal de (n+1) × (n+1), también diagonal dominante.

    Parámetros
    ----------
    t_nodes : np.ndarray, forma (n+1,)
    P : np.ndarray, forma (n+1, d) o (n+1,)
    m_start, m_end : np.ndarray, forma (d,) o escalar
        Derivadas S'(t_0) y S'(t_n).

    Retorna
    -------
    M : np.ndarray, misma forma que ``P``.

    Complejidad: O(n).
    """
    t_nodes, P2, delta, one_dim = _prepare(t_nodes, P)
    dim = P2.shape[1]
    m0 = np.broadcast_to(np.asarray(m_start, dtype=float), (dim,))
    mn = np.broadcast_to(np.asarray(m_end, dtype=float), (dim,))

    slope = np.diff(P2, axis=0) / delta[:, None]               # (n, d)
    rhs = np.empty_like(P2)
    rhs[0] = 6.0 * (slope[0] - m0)
    rhs[-1] = 6.0 * (mn - slope[-1])
    rhs[1:-1] = 6.0 * (slope[1:] - slope[:-1])

    a = np.concatenate(([0.0], delta))
    b = np.concatenate(([2.0 * delta[0]], 2.0 * (delta[:-1] + delta[1:]), [2.0 * delta[-1]]))
    c = np.concatenate((delta, [0.0]))
    M = thomas_solve(a, b, c, rhs)
    return M[:, 0] if one_dim else M


def evaluate(
    t_nodes: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    t_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evalúa el spline cúbico (natural o sujeto) y sus dos primeras derivadas.

    Con A = t_{k+1} − t y B = t − t_k::

        S(t)   = M_k·A³/(6Δ_k) + M_{k+1}·B³/(6Δ_k) + (y_k/Δ_k − M_k·Δ_k/6)·A + (y_{k+1}/Δ_k − M_{k+1}·Δ_k/6)·B
        S'(t)  = −M_k·A²/(2Δ_k) + M_{k+1}·B²/(2Δ_k) + (y_{k+1} − y_k)/Δ_k − (M_{k+1} − M_k)·Δ_k/6
        S''(t) = M_k·A/Δ_k + M_{k+1}·B/Δ_k

    Parámetros
    ----------
    t_nodes : np.ndarray, forma (n+1,)
    P : np.ndarray, forma (n+1, d) o (n+1,)
    M : np.ndarray, misma forma que ``P``
        Segundas derivadas en los nodos.
    t_eval : np.ndarray, forma (N,)

    Retorna
    -------
    (pos, d1, d2) : np.ndarray de forma (N, d), o (N,) si ``P`` es 1-D.

    Complejidad: O(N log n).
    """
    t_nodes, P2, _, one_dim = _prepare(t_nodes, P)
    M2 = np.asarray(M, dtype=float).reshape(P2.shape)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))

    k = locate_segments(t_nodes, t_eval)
    delta = (t_nodes[k + 1] - t_nodes[k])[:, None]
    A = (t_nodes[k + 1] - t_eval)[:, None]
    B = (t_eval - t_nodes[k])[:, None]
    Mk, Mk1 = M2[k], M2[k + 1]
    yk, yk1 = P2[k], P2[k + 1]

    pos = (Mk * A**3 + Mk1 * B**3) / (6 * delta) + (yk / delta - Mk * delta / 6) * A + (yk1 / delta - Mk1 * delta / 6) * B
    d1 = (-Mk * A**2 + Mk1 * B**2) / (2 * delta) + (yk1 - yk) / delta - (Mk1 - Mk) * delta / 6
    d2 = (Mk * A + Mk1 * B) / delta

    if one_dim:
        return pos[:, 0], d1[:, 0], d2[:, 0]
    return pos, d1, d2


def nodal_slopes(t_nodes: np.ndarray, P: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Primeras derivadas S'(t_k) del spline cúbico en los nodos.

    Fórmulas::

        S'(t_k) = (y_{k+1} − y_k)/Δ_k − Δ_k·(2M_k + M_{k+1})/6,           k = 0..n−1
        S'(t_n) = (y_n − y_{n−1})/Δ_{n−1} + Δ_{n−1}·(M_{n−1} + 2M_n)/6

    Sirven para dibujar las tangentes y muestran que el spline cúbico es un caso
    particular de Hermite con estas tangentes.

    Retorna
    -------
    np.ndarray, misma forma que ``P``.

    Complejidad: O(n).
    """
    t_nodes, P2, delta, one_dim = _prepare(t_nodes, P)
    M2 = np.asarray(M, dtype=float).reshape(P2.shape)
    dl = delta[:, None]
    slope = np.diff(P2, axis=0) / dl
    m = np.empty_like(P2)
    m[:-1] = slope - dl * (2 * M2[:-1] + M2[1:]) / 6
    m[-1] = slope[-1] + dl[-1] * (M2[-2] + 2 * M2[-1]) / 6
    return m[:, 0] if one_dim else m

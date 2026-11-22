"""Estimación de las tangentes ``m_k = p'(t_k)`` en los nodos para el spline de Hermite.

El spline de Hermite necesita, además de los puntos ``P_k``, la derivada de la curva
en cada nodo. Si no se conocen, se estiman a partir de los puntos vecinos.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

TangentScheme = Literal["parabolic", "central"]


def _as_2d(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    return P[:, None] if P.ndim == 1 else P


def cardinal_tangents(
    t_nodes: np.ndarray,
    P: np.ndarray,
    tension: float = 0.0,
    scheme: TangentScheme = "parabolic",
) -> np.ndarray:
    """Tangentes del spline cardinal con tensión ``c`` (``c = 0`` ⇒ Catmull-Rom).

    Con Δ_k = t_{k+1} − t_k y las pendientes de cada tramo
    d_k = (P_{k+1} − P_k)/Δ_k, en los nodos interiores (k = 1..n−1):

    * ``scheme="parabolic"`` (recomendado, Catmull-Rom no uniforme de Barry-Goldman)::

          m_k = (1 − c) · [ Δ_k·d_{k−1} + Δ_{k−1}·d_k ] / (Δ_{k−1} + Δ_k)

      Es la derivada en t_k de la parábola que pasa por P_{k−1}, P_k, P_{k+1}:
      exacta para polinomios de grado 2, con error O(h²) aun si los Δ_k son
      distintos. Es la que garantiza que la parametrización centrípeta no
      produzca bucles ni cúspides.

    * ``scheme="central"`` (fórmula clásica)::

          m_k = (1 − c) · (P_{k+1} − P_{k−1}) / (t_{k+1} − t_{k−1})

      Coincide con la anterior cuando la parametrización es uniforme, pero solo
      es de orden O(h) si los Δ_k varían.

    Extremos (diferencias unilaterales, escaladas por la tensión)::

        m_0 = (1 − c)·(P_1 − P_0)/Δ_0,      m_n = (1 − c)·(P_n − P_{n−1})/Δ_{n−1}

    Con ``c = 1`` todas las tangentes son nulas: cada tramo es un segmento recto
    recorrido con arranque y frenado suaves, y la rapidez se anula en los nodos.

    Parámetros
    ----------
    t_nodes : np.ndarray, forma (n+1,)
        Nodos estrictamente crecientes.
    P : np.ndarray, forma (n+1, d) o (n+1,)
        Waypoints.
    tension : float
        Tensión c ∈ [0, 1].
    scheme : {"parabolic", "central"}
        Fórmula para los nodos interiores.

    Retorna
    -------
    m : np.ndarray, misma forma que ``P``
        Tangentes respecto de t en cada nodo.

    Complejidad: O(n).
    """
    if not 0.0 <= tension <= 1.0:
        raise ValueError("La tensión debe estar en [0, 1].")
    if scheme not in ("parabolic", "central"):
        raise ValueError(f"Esquema de tangentes desconocido: {scheme!r}.")

    t_nodes = np.asarray(t_nodes, dtype=float)
    P_in = np.asarray(P, dtype=float)
    P2 = _as_2d(P_in)
    if P2.shape[0] != t_nodes.shape[0] or P2.shape[0] < 2:
        raise ValueError("t_nodes y P deben tener la misma cantidad (≥ 2) de nodos.")

    delta = np.diff(t_nodes)[:, None]          # (n, 1)
    d = np.diff(P2, axis=0) / delta            # pendientes de cada tramo, (n, d)

    m = np.empty_like(P2)
    m[0] = d[0]
    m[-1] = d[-1]
    if P2.shape[0] > 2:
        if scheme == "parabolic":
            m[1:-1] = (delta[1:] * d[:-1] + delta[:-1] * d[1:]) / (delta[:-1] + delta[1:])
        else:
            m[1:-1] = (P2[2:] - P2[:-2]) / (t_nodes[2:] - t_nodes[:-2])[:, None]

    m *= 1.0 - tension
    return m.reshape(P_in.shape)


def exact_tangents(P: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Valida y devuelve tangentes exactas dadas por el usuario.

    Se usan para comparar contra curvas analíticas: si p(t) es la curva exacta y
    los nodos son t_k, entonces m_k = p'(t_k) (derivada respecto del **mismo**
    parámetro t usado en ``t_nodes``).

    Parámetros
    ----------
    P : np.ndarray, forma (n+1, d)
        Waypoints (solo para verificar la forma).
    m : np.ndarray, forma (n+1, d)
        Tangentes exactas.

    Retorna
    -------
    np.ndarray
        Copia de ``m`` como arreglo de punto flotante.

    Complejidad: O(n).
    """
    m = np.array(m, dtype=float)
    if m.shape != np.shape(P):
        raise ValueError(f"Las tangentes tienen forma {m.shape} y los waypoints {np.shape(P)}.")
    if not np.all(np.isfinite(m)):
        raise ValueError("Las tangentes deben ser finitas.")
    return m

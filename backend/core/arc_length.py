"""Longitud de arco con Simpson compuesto y reparametrización por longitud de arco.

La longitud recorrida desde t_0 hasta t es::

    s(t) = ∫_{t_0}^{t} ‖p'(τ)‖ dτ

No tiene forma cerrada para un spline cúbico (la raíz de un polinomio de grado 4),
así que se integra numéricamente.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

SpeedFn = Callable[[np.ndarray], np.ndarray]


def _simpson_weights(n_sub: int) -> np.ndarray:
    """Pesos [1, 4, 2, 4, ..., 2, 4, 1] de Simpson compuesto con n_sub subintervalos."""
    w = np.ones(n_sub + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w


def simpson_composite(f: SpeedFn, a: np.ndarray | float, b: np.ndarray | float, n_sub: int = 16) -> np.ndarray:
    """Integral ∫_a^b f(τ) dτ con la regla de Simpson compuesta (vectorizada).

    Fórmula, con h = (b − a)/n_sub, τ_j = a + j·h y n_sub par::

        ∫_a^b f ≈ (h/3)·[ f(τ_0) + 4f(τ_1) + 2f(τ_2) + 4f(τ_3) + ... + 4f(τ_{n−1}) + f(τ_n) ]

    Error: −(b − a)·h⁴·f⁽⁴⁾(ξ)/180, es decir O(h⁴).

    Parámetros
    ----------
    f : callable
        Función vectorizada f(τ) → array de la misma forma.
    a, b : float o np.ndarray de forma (K,)
        Límites de integración; con arrays se calculan K integrales a la vez.
    n_sub : int
        Número de subintervalos (par, ≥ 2).

    Retorna
    -------
    np.ndarray de forma (K,) (o escalar 0-D si a y b son escalares).

    Complejidad: O(K·n_sub) evaluaciones de f.
    """
    if n_sub < 2 or n_sub % 2:
        raise ValueError("n_sub debe ser par y ≥ 2.")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_, b_ = np.broadcast_arrays(np.atleast_1d(a), np.atleast_1d(b))
    h = (b_ - a_) / n_sub
    tau = a_[:, None] + h[:, None] * np.arange(n_sub + 1)[None, :]
    vals = np.asarray(f(tau.ravel()), dtype=float).reshape(tau.shape)
    result = (h / 3.0) * (vals @ _simpson_weights(n_sub))
    return result.reshape(np.broadcast(a, b).shape)


def cumulative_arc_length(t: np.ndarray, speed_fn: SpeedFn, n_sub: int = 2) -> np.ndarray:
    """Longitud de arco acumulada s_i = s(t_i) en una malla creciente de parámetros.

    Fórmula: entre muestras consecutivas se aplica Simpson compuesto y se acumula::

        s_0 = 0,     s_{i+1} = s_i + ∫_{t_i}^{t_{i+1}} ‖p'(τ)‖ dτ

    Con n_sub = 2 cada integral usa el punto medio:
    (h/6)·[f(t_i) + 4f((t_i + t_{i+1})/2) + f(t_{i+1})].
    Si los nodos del spline están en la malla, ningún subintervalo cruza un nodo
    y el integrando es suave en cada uno, por lo que se conserva el orden O(h⁴).

    Parámetros
    ----------
    t : np.ndarray, forma (N,)
        Parámetros crecientes.
    speed_fn : callable
        τ ↦ ‖p'(τ)‖, vectorizada.
    n_sub : int
        Subintervalos de Simpson entre muestras consecutivas (par).

    Retorna
    -------
    s : np.ndarray, forma (N,)

    Complejidad: O(N·n_sub) evaluaciones de la rapidez.
    """
    t = np.asarray(t, dtype=float)
    if t.shape[0] < 2:
        return np.zeros_like(t)
    ds = simpson_composite(speed_fn, t[:-1], t[1:], n_sub)
    return np.concatenate(([0.0], np.cumsum(ds)))


def reparametrize_by_arc_length(
    t_nodes: np.ndarray,
    speed_fn: SpeedFn,
    s_targets: np.ndarray,
    n_sub: int = 64,
    tol: float = 1e-10,
    max_iter: int = 60,
) -> np.ndarray:
    """Encuentra t tal que s(t) = s* para cada distancia objetivo s* (vectorizado).

    Se resuelve F(t) = s(t) − s* = 0 con el método de Newton::

        t ← t − F(t)/F'(t),      F'(t) = ‖p'(t)‖

    con respaldo de bisección: se mantiene un intervalo [lo, hi] que contiene la
    raíz (F(lo) ≤ 0 ≤ F(hi)); si el paso de Newton sale del intervalo o la
    rapidez es casi nula, se toma el punto medio. Así converge siempre
    (bisección) y casi siempre cuadráticamente (Newton).

    s(t) se evalúa como s(t_k) + ∫_{t_k}^{t} ‖p'‖ con Simpson compuesto, donde
    s(t_k) se precalcula en los nodos.

    Uso: obtener puntos equiespaciados en distancia, por ejemplo para animar el
    dron a velocidad constante en vez de a paso constante de t.

    Parámetros
    ----------
    t_nodes : np.ndarray, forma (n+1,)
        Nodos del spline.
    speed_fn : callable
        τ ↦ ‖p'(τ)‖, vectorizada.
    s_targets : np.ndarray, forma (K,)
        Distancias objetivo en [0, L].
    n_sub : int
        Subintervalos de Simpson para cada evaluación de s(t) (s en los nodos
        usa 4·n_sub). El error de s(t) es O((Δ_k/n_sub)⁴).
    tol : float
        Tolerancia en |s(t) − s*|.
    max_iter : int
        Máximo de iteraciones.

    Retorna
    -------
    t : np.ndarray, forma (K,)

    Complejidad: O(K·n_sub·iteraciones).
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    s_targets = np.atleast_1d(np.asarray(s_targets, dtype=float))
    seg = simpson_composite(speed_fn, t_nodes[:-1], t_nodes[1:], 4 * n_sub)
    s_nodes = np.concatenate(([0.0], np.cumsum(seg)))
    s_targets = np.clip(s_targets, 0.0, s_nodes[-1])

    k = np.clip(np.searchsorted(s_nodes, s_targets, side="right") - 1, 0, t_nodes.shape[0] - 2)
    t_start = t_nodes[k]
    lo = t_nodes[k].copy()
    hi = t_nodes[k + 1].copy()
    seg_len = s_nodes[k + 1] - s_nodes[k]
    frac = np.divide(s_targets - s_nodes[k], seg_len, out=np.zeros_like(seg_len), where=seg_len > 0)
    t = lo + frac * (hi - lo)       # estimación inicial: interpolación lineal

    for _ in range(max_iter):
        F = s_nodes[k] + simpson_composite(speed_fn, t_start, t, n_sub) - s_targets
        if np.all(np.abs(F) <= tol):
            break
        lo = np.where(F < 0, t, lo)
        hi = np.where(F > 0, t, hi)
        dF = speed_fn(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_newton = t - F / dF
        ok = np.isfinite(t_newton) & (t_newton > lo) & (t_newton < hi) & (dF > 1e-14)
        t = np.where(np.abs(F) <= tol, t, np.where(ok, t_newton, 0.5 * (lo + hi)))
    return t

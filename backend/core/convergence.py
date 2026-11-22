"""Estudio de convergencia: error de interpolación frente a curvas exactas.

Se muestrean n+1 waypoints de una curva conocida p(θ), se interpola con cada
método usando como nodos el **propio parámetro** de la curva (t_k = θ_k, para que
las tangentes exactas p'(θ_k) estén en la misma variable) y se mide el error::

    E(n) = máx_θ ‖S(θ) − p(θ)‖

Si E ≈ C·h^q con h = (θ_n − θ_0)/n, el orden observado q es la pendiente de la
recta log E vs log h. Con n pequeño la curva aún no está bien resuelta (régimen
pre-asintótico), así que se reportan:

* órdenes locales entre n consecutivos: q_i = log(E_i/E_{i+1}) / log(n_{i+1}/n_i);
* orden asintótico: el último orden local (el n más fino);
* orden por ajuste: pendiente por mínimos cuadrados sobre todos los n.

Órdenes teóricos esperados (curva suave):
* Hermite con tangentes exactas:                O(h⁴)
* Spline cúbico sujeto con pendientes exactas:  O(h⁴)
* Catmull-Rom (tangentes con error O(h²)):      O(h³) en el interior; las
  diferencias unilaterales de los extremos (error O(h)) lo reducen a O(h²) cerca
  de los bordes.
* Spline natural: O(h⁴) en el interior, pero la condición M_0 = M_n = 0 es falsa
  si la curva tiene curvatura en los extremos y da O(h²) cerca de ellos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from . import cubic_natural, hermite, tangents

ConvMethod = Literal["hermite_exact", "catmull_rom", "natural", "clamped_exact"]
CurveFn = Callable[[np.ndarray], np.ndarray]

METHODS: tuple[ConvMethod, ...] = ("hermite_exact", "catmull_rom", "natural", "clamped_exact")


@dataclass(frozen=True)
class Curve:
    """Curva paramétrica exacta p(θ), θ ∈ [a, b], con su derivada p'(θ)."""

    name: str
    a: float
    b: float
    position: CurveFn
    derivative: CurveFn


def _circle_pos(theta: np.ndarray, R: float = 1.0) -> np.ndarray:
    return R * np.c_[np.cos(theta), np.sin(theta)]


def _circle_der(theta: np.ndarray, R: float = 1.0) -> np.ndarray:
    return R * np.c_[-np.sin(theta), np.cos(theta)]


def _lemniscate_pos(theta: np.ndarray, a: float = 1.0) -> np.ndarray:
    """Lemniscata de Bernoulli: x = a·cosθ/(1 + sin²θ), y = a·sinθ·cosθ/(1 + sin²θ)."""
    s, c = np.sin(theta), np.cos(theta)
    D = 1 + s**2
    return a * np.c_[c / D, s * c / D]


def _lemniscate_der(theta: np.ndarray, a: float = 1.0) -> np.ndarray:
    """Derivada de la lemniscata.

    x' = −a·sinθ·(3 − sin²θ)/D²,   y' = a·[(cos²θ − sin²θ)·D − 2sin²θ·cos²θ]/D²,   D = 1 + sin²θ
    """
    s, c = np.sin(theta), np.cos(theta)
    D = 1 + s**2
    dx = -a * s * (3 - s**2) / D**2
    dy = a * ((c**2 - s**2) * D - 2 * s**2 * c**2) / D**2
    return np.c_[dx, dy]


CIRCLE_ARC = Curve("Arco de círculo (3π/2)", 0.0, 1.5 * np.pi, _circle_pos, _circle_der)
LEMNISCATE = Curve("Lemniscata de Bernoulli", 0.0, 2 * np.pi, _lemniscate_pos, _lemniscate_der)


def interpolate_curve(curve: Curve, n: int, method: ConvMethod, theta_eval: np.ndarray) -> np.ndarray:
    """Interpola ``curve`` con n tramos (n+1 waypoints equiespaciados en θ) y evalúa en ``theta_eval``.

    Métodos:
    * ``hermite_exact``:  Hermite con m_k = p'(θ_k).
    * ``catmull_rom``:    Hermite con tangentes cardinales (c = 0).
    * ``natural``:        spline cúbico natural (M_0 = M_n = 0).
    * ``clamped_exact``:  spline cúbico sujeto con S'(θ_0) = p'(θ_0), S'(θ_n) = p'(θ_n).

    Retorna
    -------
    np.ndarray, forma (N, 2): posición interpolada.

    Complejidad: O(n + N log n).
    """
    t_nodes = np.linspace(curve.a, curve.b, n + 1)
    P = curve.position(t_nodes)
    if method == "hermite_exact":
        m = tangents.exact_tangents(P, curve.derivative(t_nodes))
        return hermite.evaluate(t_nodes, P, m, theta_eval)[0]
    if method == "catmull_rom":
        m = tangents.cardinal_tangents(t_nodes, P, 0.0)
        return hermite.evaluate(t_nodes, P, m, theta_eval)[0]
    if method == "natural":
        M = cubic_natural.natural_second_derivatives(t_nodes, P)
        return cubic_natural.evaluate(t_nodes, P, M, theta_eval)[0]
    if method == "clamped_exact":
        d = curve.derivative(np.array([curve.a, curve.b]))
        M = cubic_natural.clamped_second_derivatives(t_nodes, P, d[0], d[1])
        return cubic_natural.evaluate(t_nodes, P, M, theta_eval)[0]
    raise ValueError(f"Método desconocido: {method!r}")


def observed_order(n_values: np.ndarray, errors: np.ndarray) -> float:
    """Orden observado q: pendiente de log E frente a log h con h ∝ 1/n.

    Fórmula (mínimos cuadrados sobre los pares (log h_i, log E_i))::

        log E ≈ q·log h + log C,      q = pendiente de la recta ajustada

    Complejidad: O(len(n_values)).
    """
    h = 1.0 / np.asarray(n_values, dtype=float)
    return float(np.polyfit(np.log(h), np.log(np.asarray(errors, dtype=float)), 1)[0])


def local_orders(n_values: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Órdenes locales entre pares consecutivos de n.

    Fórmula::

        q_i = log(E_i / E_{i+1}) / log(n_{i+1} / n_i)

    Complejidad: O(len(n_values)).
    """
    n = np.asarray(n_values, dtype=float)
    e = np.asarray(errors, dtype=float)
    return np.log(e[:-1] / e[1:]) / np.log(n[1:] / n[:-1])


def convergence_study(
    curve: Curve,
    n_values: tuple[int, ...] = (4, 8, 16, 32, 64),
    methods: tuple[ConvMethod, ...] = METHODS,
    n_eval: int = 8001,
    interior: tuple[float, float] = (0.25, 0.75),
) -> dict[str, dict[str, np.ndarray | float]]:
    """Error máximo de posición para cada método y cada n, con su orden observado.

    Además del error global se mide el error en una ventana interior de θ
    (por defecto el 50 % central), para separar el efecto de las condiciones de
    frontera del error propio del esquema.

    Retorna
    -------
    dict método → {"n", "error", "error_interior", "local_orders",
    "local_orders_interior", "order", "order_interior", "order_fit"}, donde
    "order" y "order_interior" son los órdenes asintóticos (último par de n).

    Complejidad: O(len(n_values)·len(methods)·n_eval).
    """
    theta = np.linspace(curve.a, curve.b, n_eval)
    exact = curve.position(theta)
    lo = curve.a + interior[0] * (curve.b - curve.a)
    hi = curve.a + interior[1] * (curve.b - curve.a)
    inner = (theta >= lo) & (theta <= hi)
    n_arr = np.asarray(n_values)

    out: dict[str, dict[str, np.ndarray | float]] = {}
    for method in methods:
        err = np.empty(len(n_arr))
        err_in = np.empty(len(n_arr))
        for i, n in enumerate(n_arr):
            dist = np.linalg.norm(interpolate_curve(curve, int(n), method, theta) - exact, axis=1)
            err[i] = dist.max()
            err_in[i] = dist[inner].max()
        q = local_orders(n_arr, err)
        q_in = local_orders(n_arr, err_in)
        out[method] = {
            "n": n_arr,
            "error": err,
            "error_interior": err_in,
            "local_orders": q,
            "local_orders_interior": q_in,
            "order": float(q[-1]),
            "order_interior": float(q_in[-1]),
            "order_fit": observed_order(n_arr, err),
        }
    return out

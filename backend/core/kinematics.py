"""Magnitudes cinemáticas de una curva paramétrica p(t) = (x(t), y(t)).

Importante: ``speed`` y ``acceleration_norm`` son derivadas respecto del
**parámetro** t, no del tiempo físico. Sus unidades dependen de la
parametrización (con la de cuerda, t ≈ metros y ‖p'‖ ≈ 1). La curvatura κ, en
cambio, es geométrica: no depende de cómo se parametrice la curva.
La velocidad física del dron es el perfil de ``velocity_profile``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

#: Curvatura máxima reportada (radio de giro mínimo de 1 µm). Se usa para
#: reemplazar κ = ∞ en puntos singulares, donde la curva tiene una esquina.
KAPPA_MAX: float = 1e6


def speed(d1: np.ndarray) -> np.ndarray:
    """Rapidez paramétrica ‖p'(t)‖.

    Fórmula::

        ‖p'(t)‖ = √(x'(t)² + y'(t)²)

    Parámetros
    ----------
    d1 : np.ndarray, forma (N, d)
        Primera derivada respecto de t.

    Retorna
    -------
    np.ndarray, forma (N,)

    Complejidad: O(N).
    """
    return np.linalg.norm(np.asarray(d1, dtype=float), axis=-1)


def acceleration_norm(d2: np.ndarray) -> np.ndarray:
    """Norma de la aceleración paramétrica ‖p''(t)‖.

    Fórmula::

        ‖p''(t)‖ = √(x''(t)² + y''(t)²)

    Es la magnitud que delata la continuidad: en Hermite (C¹) salta en los nodos,
    en el spline cúbico natural (C²) es continua.

    Complejidad: O(N).
    """
    return np.linalg.norm(np.asarray(d2, dtype=float), axis=-1)


def curvature(d1: np.ndarray, d2: np.ndarray, eps: float = 1e-12, kappa_max: float = KAPPA_MAX) -> np.ndarray:
    """Curvatura κ de una curva plana.

    Fórmula::

        κ = |x'·y'' − y'·x''| / (x'² + y'²)^{3/2}

    El radio de curvatura es 1/κ. Si la rapidez paramétrica es casi nula
    (‖p'‖ ≤ eps, por ejemplo en los nodos con tensión c = 1) la fórmula es 0/0:
    la curva tiene una esquina y se devuelve ``kappa_max`` (giro "en el lugar",
    lo que obliga a detener el dron). El resultado se acota por ``kappa_max``
    para que siempre sea finito.

    Parámetros
    ----------
    d1, d2 : np.ndarray, forma (N, 2)
        Primera y segunda derivada respecto de t.
    eps : float
        Umbral de rapidez considerada nula.
    kappa_max : float
        Cota superior de κ.

    Retorna
    -------
    kappa : np.ndarray, forma (N,), valores en [0, kappa_max].

    Complejidad: O(N).
    """
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    v = speed(d1)
    singular = v <= eps
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(singular, kappa_max, cross / np.where(singular, 1.0, v) ** 3)
    return np.minimum(kappa, kappa_max)


GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0   # 1/φ ≈ 0.618


def curvature_peaks(
    kappa_fn: Callable[[np.ndarray], np.ndarray],
    t: np.ndarray,
    kappa: np.ndarray,
    iters: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Ubica con precisión los máximos de κ(t) que caen entre las muestras.

    En curvas cerradas (rapidez paramétrica casi nula) κ tiene picos muy angostos,
    y una malla uniforme los subestima (hasta en un 99 % con 50 muestras por tramo).
    Para cada máximo local de la muestra κ_i (κ_i ≥ κ_{i−1} y κ_i ≥ κ_{i+1}) se
    busca el máximo en [t_{i−1}, t_{i+1}] por **sección áurea**::

        c = b − (b − a)/φ,   d = a + (b − a)/φ
        si κ(c) > κ(d):  b ← d    si no:  a ← c

    El intervalo se reduce por 1/φ ≈ 0.618 en cada iteración (60 iteraciones ⇒
    factor 3·10⁻¹³). Todos los picos se refinan a la vez (vectorizado).

    Parámetros
    ----------
    kappa_fn : callable
        t ↦ κ(t), vectorizada.
    t : np.ndarray, forma (N,)
        Parámetros de las muestras (crecientes).
    kappa : np.ndarray, forma (N,)
        κ en las muestras.
    iters : int
        Iteraciones de sección áurea.

    Retorna
    -------
    (t_peak, kappa_peak) : np.ndarray de forma (K,)
        Posición y valor de cada pico (κ_peak ≥ κ de la muestra que lo originó).

    Complejidad: O(K·iters) evaluaciones de κ, con K = número de máximos locales.
    """
    t = np.asarray(t, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    N = t.shape[0]
    if N < 2:
        return np.empty(0), np.empty(0)
    left = np.concatenate(([-np.inf], kappa[:-1]))
    right = np.concatenate((kappa[1:], [-np.inf]))
    idx = np.flatnonzero((kappa >= left) & (kappa >= right))
    a = t[np.maximum(idx - 1, 0)]
    b = t[np.minimum(idx + 1, N - 1)]

    c = b - GOLDEN * (b - a)
    d = a + GOLDEN * (b - a)
    fc, fd = kappa_fn(c), kappa_fn(d)
    # Mejor punto visto: necesario porque en Hermite κ salta en los nodos (C¹) y el
    # máximo puede ser un límite lateral que el punto final de la búsqueda no alcanza.
    best_t = np.where(fc >= fd, c, d)
    best_k = np.maximum(fc, fd)
    for _ in range(iters):
        keep_left = fc > fd                       # el máximo está en [a, d]
        a = np.where(keep_left, a, c)
        b = np.where(keep_left, d, b)
        new = np.where(keep_left, b - GOLDEN * (b - a), a + GOLDEN * (b - a))
        f_new = kappa_fn(new)
        better = f_new > best_k
        best_t = np.where(better, new, best_t)
        best_k = np.where(better, f_new, best_k)
        c, d = np.where(keep_left, new, d), np.where(keep_left, c, new)
        fc, fd = np.where(keep_left, f_new, fd), np.where(keep_left, fc, f_new)
    sampled = kappa[idx] >= best_k
    t_peak = np.where(sampled, t[idx], best_t)
    kappa_peak = np.maximum(best_k, kappa[idx])
    return t_peak, kappa_peak


def physical_accelerations(s: np.ndarray, v: np.ndarray, kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aceleraciones físicas del dron que recorre la curva con el perfil v(s).

    Fórmulas::

        a_lat  = v²·κ                    (aceleración centrípeta o lateral)
        a_long = v·dv/ds = d(v²/2)/ds    (aceleración tangencial)

    a_long se aproxima con diferencias finitas de v²/2 respecto de s
    (``np.gradient``, de segundo orden en los puntos interiores).

    Parámetros
    ----------
    s : np.ndarray, forma (N,)
        Longitud de arco acumulada.
    v : np.ndarray, forma (N,)
        Velocidad física en cada punto.
    kappa : np.ndarray, forma (N,)
        Curvatura en cada punto.

    Retorna
    -------
    (a_lat, a_long) : np.ndarray de forma (N,).

    Complejidad: O(N).
    """
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    a_lat = v**2 * np.asarray(kappa, dtype=float)
    if s.shape[0] < 2:
        return a_lat, np.zeros_like(v)
    a_long = np.gradient(0.5 * v**2, s)
    return a_lat, a_long

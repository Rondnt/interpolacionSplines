"""Fase 3 — Algoritmo de Thomas y spline cúbico natural/sujeto."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from core import cubic_natural, hermite, parametrization

WAYPOINTS = np.array([[0.0, 0.0], [2.0, 3.0], [5.0, 1.0], [7.0, 4.0], [9.0, 4.5], [10.0, 1.0]])


def _tridiagonal_aleatorio(rng, N):
    a = rng.uniform(-1, 1, N)
    c = rng.uniform(-1, 1, N)
    a[0] = c[-1] = 0.0
    b = np.abs(a) + np.abs(c) + rng.uniform(0.5, 2.0, N)    # diagonal dominante
    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    return a, b, c, A


@pytest.mark.parametrize("N", [1, 2, 5, 50, 500])
def test_thomas_coincide_con_numpy(N):
    rng = np.random.default_rng(N)
    a, b, c, A = _tridiagonal_aleatorio(rng, N)
    d = rng.normal(size=N)
    assert np.allclose(cubic_natural.thomas_solve(a, b, c, d), np.linalg.solve(A, d), atol=1e-12)


def test_thomas_varias_columnas():
    rng = np.random.default_rng(0)
    a, b, c, A = _tridiagonal_aleatorio(rng, 20)
    d = rng.normal(size=(20, 3))
    assert np.allclose(cubic_natural.thomas_solve(a, b, c, d), np.linalg.solve(A, d), atol=1e-12)


@pytest.mark.parametrize("mode", ["uniform", "chord", "centripetal"])
def test_natural_coincide_con_scipy(mode):
    t = parametrization.parametrize(WAYPOINTS, mode)
    M = cubic_natural.natural_second_derivatives(t, WAYPOINTS)
    t_eval = np.linspace(t[0], t[-1], 2001)
    pos, d1, d2 = cubic_natural.evaluate(t, WAYPOINTS, M, t_eval)
    ref = CubicSpline(t, WAYPOINTS, bc_type="natural")
    assert np.max(np.abs(pos - ref(t_eval))) < 1e-10
    assert np.max(np.abs(d1 - ref(t_eval, 1))) < 1e-10
    assert np.max(np.abs(d2 - ref(t_eval, 2))) < 1e-9
    assert np.allclose(M[[0, -1]], 0.0)


@pytest.mark.parametrize("mode", ["uniform", "chord", "centripetal"])
def test_sujeto_coincide_con_scipy(mode):
    t = parametrization.parametrize(WAYPOINTS, mode)
    m0, mn = np.array([1.0, -2.0]), np.array([0.5, 0.5])
    M = cubic_natural.clamped_second_derivatives(t, WAYPOINTS, m0, mn)
    t_eval = np.linspace(t[0], t[-1], 2001)
    pos, d1, d2 = cubic_natural.evaluate(t, WAYPOINTS, M, t_eval)
    ref = CubicSpline(t, WAYPOINTS, bc_type=((1, m0), (1, mn)))
    assert np.max(np.abs(pos - ref(t_eval))) < 1e-10
    assert np.max(np.abs(d1 - ref(t_eval, 1))) < 1e-10
    assert np.max(np.abs(d2 - ref(t_eval, 2))) < 1e-9


def test_dos_puntos_es_recta():
    P = np.array([[0.0, 0.0], [3.0, 4.0]])
    t = parametrization.parametrize(P, "chord")
    M = cubic_natural.natural_second_derivatives(t, P)
    pos, _, d2 = cubic_natural.evaluate(t, P, M, np.linspace(0, 5, 11))
    assert np.allclose(pos, np.linspace(0, 1, 11)[:, None] * P[1])
    assert np.allclose(d2, 0.0)


@pytest.mark.parametrize("bc", ["natural", "clamped"])
def test_continuidad_C2(bc):
    t = parametrization.parametrize(WAYPOINTS, "centripetal")
    if bc == "natural":
        M = cubic_natural.natural_second_derivatives(t, WAYPOINTS)
    else:
        M = cubic_natural.clamped_second_derivatives(t, WAYPOINTS, [1.0, 0.0], [0.0, 1.0])
    eps = 1e-9
    tk = t[1:-1]
    left = cubic_natural.evaluate(t, WAYPOINTS, M, tk - eps)
    right = cubic_natural.evaluate(t, WAYPOINTS, M, tk + eps)
    for order, tol in zip(range(3), (1e-7, 1e-6, 1e-5)):
        assert np.max(np.abs(right[order] - left[order])) < tol
    pos, _, _ = cubic_natural.evaluate(t, WAYPOINTS, M, t)
    assert np.max(np.abs(pos - WAYPOINTS)) < 1e-12


def test_cubico_es_hermite_con_sus_pendientes():
    """El spline cúbico es un Hermite cuyas tangentes son S'(t_k)."""
    t = parametrization.parametrize(WAYPOINTS, "chord")
    M = cubic_natural.natural_second_derivatives(t, WAYPOINTS)
    m = cubic_natural.nodal_slopes(t, WAYPOINTS, M)
    t_eval = np.linspace(t[0], t[-1], 500)
    a = cubic_natural.evaluate(t, WAYPOINTS, M, t_eval)[0]
    b = hermite.evaluate(t, WAYPOINTS, m, t_eval)[0]
    assert np.allclose(a, b, atol=1e-10)
    assert np.allclose(hermite.second_derivative_jumps(t, WAYPOINTS, m), 0.0, atol=1e-9)

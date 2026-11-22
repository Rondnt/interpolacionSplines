"""Fase 4 — Longitud de arco (Simpson) y reparametrización."""

import numpy as np
import pytest
from scipy.integrate import quad

from core import arc_length, hermite, parametrization, tangents, trajectory

WAYPOINTS = np.array([[0.0, 0.0], [2.0, 3.0], [5.0, 1.0], [7.0, 4.0], [9.0, 4.5], [10.0, 1.0]])


def _spline():
    t = parametrization.parametrize(WAYPOINTS, "centripetal")
    m = tangents.cardinal_tangents(t, WAYPOINTS)

    def speed(tau):
        return np.linalg.norm(hermite.evaluate(t, WAYPOINTS, m, tau)[1], axis=1)

    return t, speed


def _s_reference(t_nodes, speed, t_values):
    """s(t) de referencia con scipy.quad, partiendo la integral en los nodos (donde ‖p'‖ tiene un quiebre)."""
    out = []
    for ti in t_values:
        breaks = t_nodes[(t_nodes > t_nodes[0]) & (t_nodes < ti)]
        val, _ = quad(lambda x: float(speed(np.array([x]))[0]), t_nodes[0], ti,
                      points=breaks if len(breaks) else None, epsabs=1e-13, epsrel=1e-13, limit=200)
        out.append(val)
    return np.array(out)


def test_simpson_exacto_para_cubicas():
    f = lambda x: 4 * x**3 - 3 * x**2 + 2      # noqa: E731
    assert np.isclose(arc_length.simpson_composite(f, 0.0, 2.0, 2), 16 - 8 + 4)


def test_simpson_orden_4():
    errs = [abs(arc_length.simpson_composite(np.sin, 0.0, np.pi, n) - 2.0) for n in (8, 16, 32)]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(orders - 4) < 0.1)


def test_simpson_coincide_con_quad_en_un_tramo():
    t, speed = _spline()
    for k in range(len(t) - 1):
        ref, _ = quad(lambda x: float(speed(np.array([x]))[0]), t[k], t[k + 1], epsabs=1e-13)
        val = float(arc_length.simpson_composite(speed, t[k], t[k + 1], 256))
        assert abs(val - ref) < 1e-9 * max(ref, 1.0)


def test_longitud_circulo():
    """Longitud de un círculo de radio R muestreado con 64 waypoints ≈ 2πR (error < 1e-3)."""
    R = 2.0
    theta = np.linspace(0, 2 * np.pi, 65)
    P = R * np.c_[np.cos(theta), np.sin(theta)]
    res = trajectory.compute_trajectory(P, "hermite", 0.0, "chord", samples_per_segment=20)
    err = abs(res.length - 2 * np.pi * R) / (2 * np.pi * R)
    print(f"\nError relativo de la longitud del círculo: {err:.2e}")
    assert err < 1e-3


def test_longitud_recta_exacta():
    P = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    res = trajectory.compute_trajectory(P, "hermite", 0.0, "chord")
    assert np.isclose(res.length, 10.0, rtol=1e-12)


@pytest.mark.parametrize("n_targets", [1, 50, 400])
def test_reparametrizacion_equiespaciada(n_targets):
    t, speed = _spline()
    L = float(_s_reference(t, speed, [t[-1]])[0])
    s_targets = np.linspace(0, L, n_targets)
    t_s = arc_length.reparametrize_by_arc_length(t, speed, s_targets)
    # s(t_s) reconstruido de forma independiente con scipy.quad
    s_check = _s_reference(t, speed, t_s)
    assert np.max(np.abs(s_check - s_targets)) < 1e-6
    assert np.all(np.diff(t_s) > 0) or n_targets == 1


def test_reparametrizacion_con_velocidad_nula_en_nodos():
    """Con tensión 1 la rapidez se anula en los nodos: la bisección de respaldo converge igual."""
    t = parametrization.parametrize(WAYPOINTS, "uniform")
    m = tangents.cardinal_tangents(t, WAYPOINTS, 1.0)

    def speed(tau):
        return np.linalg.norm(hermite.evaluate(t, WAYPOINTS, m, tau)[1], axis=1)

    total = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1).sum()   # polilínea
    s_targets = np.linspace(0, total, 37)
    t_s = arc_length.reparametrize_by_arc_length(t, speed, s_targets)
    s_check = _s_reference(t, speed, t_s)
    assert np.max(np.abs(s_check - s_targets)) < 1e-6

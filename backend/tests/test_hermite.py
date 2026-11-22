"""Fase 2 — Parametrización, tangentes y spline de Hermite."""

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline

from core import hermite, parametrization, tangents

WAYPOINTS = np.array([[0.0, 0.0], [2.0, 3.0], [5.0, 1.0], [7.0, 4.0], [9.0, 4.5], [10.0, 1.0]])


# --------------------------------------------------------------------------- parametrización
def test_parametrizacion_uniforme():
    assert np.array_equal(parametrization.parametrize(WAYPOINTS, "uniform"), np.arange(6.0))


def test_parametrizacion_cuerda_y_centripeta():
    dist = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1)
    t_chord = parametrization.parametrize(WAYPOINTS, "chord")
    t_centr = parametrization.parametrize(WAYPOINTS, "centripetal")
    assert np.allclose(np.diff(t_chord), dist)
    assert np.allclose(np.diff(t_centr), np.sqrt(dist))
    assert t_chord[0] == t_centr[0] == 0.0


def test_parametrizacion_rechaza_duplicados():
    P = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="coinciden"):
        parametrization.parametrize(P, "chord")


# --------------------------------------------------------------------------- tangentes
@pytest.mark.parametrize("mode", ["uniform", "chord", "centripetal"])
def test_tangente_parabolica_exacta_en_cuadraticas(mode):
    """La tangente parabólica es exacta para curvas de grado 2 en los nodos interiores."""
    t = parametrization.parametrize(WAYPOINTS, mode)
    P = np.c_[t**2 - 3 * t, 0.5 * t**2 + t]     # curva cuadrática muestreada en t
    exact = np.c_[2 * t - 3, t + 1]
    m = tangents.cardinal_tangents(t, P, scheme="parabolic")
    assert np.allclose(m[1:-1], exact[1:-1], atol=1e-12)


def test_tangentes_central_y_parabolica_coinciden_en_uniforme():
    t = np.arange(6.0)
    m_par = tangents.cardinal_tangents(t, WAYPOINTS, scheme="parabolic")
    m_cen = tangents.cardinal_tangents(t, WAYPOINTS, scheme="central")
    assert np.allclose(m_par, m_cen)


def test_tension_escala_tangentes():
    t = parametrization.parametrize(WAYPOINTS, "chord")
    m0 = tangents.cardinal_tangents(t, WAYPOINTS, 0.0)
    m5 = tangents.cardinal_tangents(t, WAYPOINTS, 0.5)
    m1 = tangents.cardinal_tangents(t, WAYPOINTS, 1.0)
    assert np.allclose(m5, 0.5 * m0)
    assert np.allclose(m1, 0.0)


def test_tension_fuera_de_rango():
    with pytest.raises(ValueError):
        tangents.cardinal_tangents(np.arange(6.0), WAYPOINTS, 1.5)


# --------------------------------------------------------------------------- bases
def test_bases_particion_y_valores_extremos():
    u = np.linspace(0, 1, 101)
    h00, h10, h01, h11 = hermite.hermite_basis(u)
    assert np.allclose(h00 + h01, 1.0)       # reproduce constantes
    assert hermite.hermite_basis(0.0) == (1.0, 0.0, 0.0, 0.0)
    assert hermite.hermite_basis(1.0) == (0.0, 0.0, 1.0, 0.0)
    d = hermite.hermite_basis_d1(np.array([0.0, 1.0]))
    assert np.allclose(np.array(d)[:, 0], [0, 1, 0, 0])
    assert np.allclose(np.array(d)[:, 1], [0, 0, 0, 1])


# --------------------------------------------------------------------------- spline
@pytest.mark.parametrize("mode", ["uniform", "chord", "centripetal"])
def test_interpola_waypoints(mode):
    t = parametrization.parametrize(WAYPOINTS, mode)
    m = tangents.cardinal_tangents(t, WAYPOINTS, 0.3)
    pos, d1, _ = hermite.evaluate(t, WAYPOINTS, m, t)
    assert np.max(np.abs(pos - WAYPOINTS)) < 1e-12
    assert np.max(np.abs(d1 - m)) < 1e-12


@pytest.mark.parametrize("mode", ["uniform", "chord", "centripetal"])
def test_coincide_con_scipy(mode):
    t = parametrization.parametrize(WAYPOINTS, mode)
    m = tangents.cardinal_tangents(t, WAYPOINTS, 0.0)
    t_eval = np.linspace(t[0], t[-1], 2001)
    pos, d1, d2 = hermite.evaluate(t, WAYPOINTS, m, t_eval)
    ref = CubicHermiteSpline(t, WAYPOINTS, m)
    assert np.max(np.abs(pos - ref(t_eval))) < 1e-10
    assert np.max(np.abs(d1 - ref(t_eval, 1))) < 1e-10
    assert np.max(np.abs(d2 - ref(t_eval, 2))) < 1e-9


def test_evaluate_1d():
    t = np.arange(4.0)
    y = np.array([0.0, 1.0, 0.0, 2.0])
    m = tangents.cardinal_tangents(t, y)
    pos, d1, d2 = hermite.evaluate(t, y, m, np.array([0.0, 1.5, 3.0]))
    assert pos.shape == d1.shape == d2.shape == (3,)
    assert pos[0] == 0.0 and pos[-1] == 2.0


def test_continuidad_C1_y_salto_C2():
    """p y p' son continuas en los nodos; p'' en general salta (se mide y reporta)."""
    t = parametrization.parametrize(WAYPOINTS, "centripetal")
    m = tangents.cardinal_tangents(t, WAYPOINTS)
    eps = 1e-9
    tk = t[1:-1]
    p_l, d1_l, d2_l = hermite.evaluate(t, WAYPOINTS, m, tk - eps)
    p_r, d1_r, d2_r = hermite.evaluate(t, WAYPOINTS, m, tk + eps)
    assert np.max(np.abs(p_r - p_l)) < 1e-7
    assert np.max(np.abs(d1_r - d1_l)) < 1e-6

    jumps = hermite.second_derivative_jumps(t, WAYPOINTS, m)
    assert np.allclose(jumps, d2_r - d2_l, atol=1e-5)
    print("\nSalto de p'' en los nodos interiores (Hermite):", np.linalg.norm(jumps, axis=1).round(4))
    assert np.linalg.norm(jumps, axis=1).max() > 1e-3   # no es C²

"""Fase 4 — Perfil de velocidad y tiempo estimado."""

import numpy as np
import pytest

from core import kinematics, trajectory, velocity_profile

WAYPOINTS = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [8.0, 3.0], [10.0, 6.0], [14.0, 6.0]])
LIMITS = dict(v_cruise=5.0, a_lat_max=2.0, a_long_max=1.5)


def _bucle_secuencial(v_max, s, a, v0=0.0, vn=0.0):
    """Implementación de referencia con bucles (para validar la versión vectorizada)."""
    v = v_max.copy()
    v[0] = min(v[0], v0)
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a * (s[i] - s[i - 1])))
    v[-1] = min(v[-1], vn)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a * (s[i + 1] - s[i])))
    return v


def test_limite_por_curvatura():
    kappa = np.array([0.0, 0.01, 0.5, 2.0, kinematics.KAPPA_MAX])
    v = velocity_profile.curvature_speed_limit(kappa, 5.0, 2.0)
    assert np.allclose(v, [5.0, 5.0, 2.0, 1.0, np.sqrt(2e-6)])


@pytest.mark.parametrize("seed", range(5))
def test_longitudinal_vectorizado_igual_a_bucle(seed):
    rng = np.random.default_rng(seed)
    s = np.cumsum(rng.uniform(0.01, 0.5, 300))
    s -= s[0]
    v_max = rng.uniform(0.0, 6.0, 300)
    v = velocity_profile.longitudinal_limit(v_max, s, 1.5)
    assert np.allclose(v, _bucle_secuencial(v_max, s, 1.5), atol=1e-12)


@pytest.mark.parametrize("method", ["hermite", "natural", "clamped"])
def test_perfil_respeta_limites(method):
    res = trajectory.compute_trajectory(WAYPOINTS, method, 0.0, "centripetal", **LIMITS)
    v, kappa, s = res.v_profile, res.curvature, res.s
    tol = 1e-9
    assert np.all(v <= LIMITS["v_cruise"] + tol)
    with np.errstate(divide="ignore"):
        v_lat = np.where(kappa > 0, np.sqrt(LIMITS["a_lat_max"] / kappa), np.inf)
    assert np.all(v <= v_lat + tol)
    dv2 = np.abs(np.diff(v**2))
    assert np.all(dv2 <= 2 * LIMITS["a_long_max"] * np.diff(s) + tol)
    assert v[0] == 0.0 and v[-1] == 0.0
    assert np.all(np.isfinite(res.time)) and res.time_estimate > 0


def test_tiempo_recta_acelerar_y_frenar():
    """Recta de L m sin alcanzar crucero: T = 2·√(L/a) (acelera la mitad y frena la otra)."""
    L, a = 10.0, 1.5
    res = trajectory.compute_trajectory(np.array([[0.0, 0.0], [L, 0.0]]), samples_per_segment=400,
                                        v_cruise=100.0, a_lat_max=2.0, a_long_max=a)
    assert np.isclose(res.time_estimate, 2 * np.sqrt(L / a), rtol=1e-6)


def test_tiempo_recta_con_crucero():
    """Recta larga: acelera hasta v_c, crucero y frena. T = L/v_c + v_c/a."""
    L, a, vc = 50.0, 1.5, 5.0
    res = trajectory.compute_trajectory(np.array([[0.0, 0.0], [L, 0.0]]), samples_per_segment=2000,
                                        v_cruise=vc, a_lat_max=2.0, a_long_max=a)
    assert np.isclose(res.time_estimate, L / vc + vc / a, rtol=1e-3)


def test_tension_1_no_produce_nan():
    res = trajectory.compute_trajectory(WAYPOINTS, "hermite", 1.0, "uniform", **LIMITS)
    for arr in (res.curvature, res.v_profile, res.time, res.s):
        assert np.all(np.isfinite(arr))
    # el dron se detiene (casi) en cada esquina
    assert np.all(res.v_profile[res.knots_index] < 1e-2)


@pytest.mark.parametrize("method", ["hermite", "natural"])
@pytest.mark.parametrize("seed", range(6))
def test_limite_lateral_tambien_entre_muestras(method, seed):
    """Con v interpolada linealmente en s, v²κ ≤ a_lat en una malla 40 veces más fina."""
    from core import arc_length

    rng = np.random.default_rng(seed)
    while True:
        P = np.c_[rng.uniform(0, 24, 7), rng.uniform(0, 14, 7)]
        if np.linalg.norm(np.diff(P, axis=0), axis=1).min() > 0.5:
            break
    res = trajectory.compute_trajectory(P, method, 0.0, "centripetal", samples_per_segment=50, **LIMITS)
    sp = trajectory.build_spline(P, method, 0.0, "centripetal")
    t_fine, _ = trajectory.sample_parameters(sp.t_nodes, 2000)
    s_fine = arc_length.cumulative_arc_length(t_fine, sp.speed)
    v_fine = np.interp(s_fine, res.s, res.v_profile)
    a_lat = v_fine**2 * sp.curvature(t_fine)
    assert a_lat.max() <= LIMITS["a_lat_max"] * (1 + 1e-6)

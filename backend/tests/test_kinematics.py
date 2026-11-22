"""Fase 4 — Cinemática: rapidez, aceleración y curvatura."""

import numpy as np

from core import hermite, kinematics, trajectory

R = 3.0


def _circulo_exacto(theta):
    """Posición y derivadas analíticas del círculo de radio R parametrizado por θ."""
    c, s = np.cos(theta), np.sin(theta)
    return R * np.c_[c, s], R * np.c_[-s, c], R * np.c_[-c, -s]


def test_curvatura_con_derivadas_analiticas():
    theta = np.linspace(0, 2 * np.pi, 500)
    _, d1, d2 = _circulo_exacto(theta)
    assert np.allclose(kinematics.curvature(d1, d2), 1 / R, atol=1e-12)
    assert np.allclose(kinematics.speed(d1), R)
    assert np.allclose(kinematics.acceleration_norm(d2), R)


def test_curvatura_parabola():
    """y = x²: κ(x) = 2 / (1 + 4x²)^{3/2}."""
    x = np.linspace(-2, 2, 101)
    d1 = np.c_[np.ones_like(x), 2 * x]
    d2 = np.c_[np.zeros_like(x), 2 * np.ones_like(x)]
    assert np.allclose(kinematics.curvature(d1, d2), 2 / (1 + 4 * x**2) ** 1.5)


def test_curvatura_recta_es_cero_y_singular_es_finita():
    d1 = np.array([[1.0, 2.0], [0.0, 0.0]])
    d2 = np.array([[2.0, 4.0], [1.0, 0.0]])
    kappa = kinematics.curvature(d1, d2)
    assert kappa[0] == 0.0
    assert kappa[1] == kinematics.KAPPA_MAX
    assert np.all(np.isfinite(kappa))


def test_curvatura_spline_circulo_tangentes_exactas():
    """Hermite con tangentes exactas sobre un círculo: κ ≈ 1/R (error relativo < 1e-3)."""
    n = 100
    theta_k = np.linspace(0, 2 * np.pi, n + 1)
    P, m, _ = _circulo_exacto(theta_k)
    theta = np.linspace(0, 2 * np.pi, 20001)
    _, d1, d2 = hermite.evaluate(theta_k, P, m, theta)
    kappa = kinematics.curvature(d1, d2)
    err = np.max(np.abs(kappa * R - 1))
    print(f"\nError relativo maximo de kappa (n={n}): {err:.2e}")
    assert err < 1e-3


def test_curvatura_circulo_pipeline_catmull_rom():
    """Con tangentes estimadas, lejos de los extremos κ ≈ 1/R."""
    theta_k = np.linspace(0, 2 * np.pi, 201)
    P = R * np.c_[np.cos(theta_k), np.sin(theta_k)]
    res = trajectory.compute_trajectory(P, "hermite", 0.0, "chord", samples_per_segment=20)
    n_seg = len(theta_k) - 1
    interior = slice(20 * n_seg // 10, 20 * 9 * n_seg // 10)
    err = np.max(np.abs(res.curvature[interior] * R - 1))
    print(f"Error relativo de kappa en el interior (Catmull-Rom, 200 tramos): {err:.2e}")
    assert err < 1e-2


def test_aceleraciones_fisicas():
    s = np.linspace(0, 10, 101)
    v = np.sqrt(2 * 1.5 * s)          # aceleración constante a = 1.5 desde el reposo
    kappa = np.full_like(s, 0.5)
    a_lat, a_long = kinematics.physical_accelerations(s, v, kappa)
    assert np.allclose(a_lat, v**2 * 0.5)
    assert np.allclose(a_long, 1.5)

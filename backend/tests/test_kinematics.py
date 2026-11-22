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


def test_picos_angostos_entre_muestras():
    """La sección áurea encuentra, por intervalo, un pico más angosto que la malla."""
    t = np.linspace(0, 1, 101)                      # paso 0.01
    f = lambda x: 1.0 / ((x - 0.5037) ** 2 + 1e-6)  # noqa: E731  pico de altura 1e6 y ancho ~1e-3
    k_int = kinematics.interval_max_curvature(f, t, f(t))
    assert k_int.shape == (100,)
    assert np.argmax(k_int) == 50                   # el pico está en [0.50, 0.51]
    assert np.isclose(k_int[50], 1e6, rtol=1e-6)
    assert f(t).max() < 0.2 * 1e6                   # la malla sola lo subestimaba
    # en los demás intervalos f es monótona: el máximo es el mayor extremo
    ext = np.maximum(f(t)[:-1], f(t)[1:])
    mask = np.arange(100) != 50
    assert np.allclose(k_int[mask], ext[mask], rtol=1e-12)


def test_max_curvature_incluye_picos_en_nodos_de_hermite():
    """En Hermite κ salta en los nodos: el máximo puede ser el límite por la izquierda."""
    P = np.array([[0.0, 0.0], [6.0, 0.4], [6.3, 1.2], [1.0, 2.0], [7.0, 3.0]])
    res = trajectory.compute_trajectory(P, "hermite", 0.0, "centripetal", samples_per_segment=50)
    sp = trajectory.build_spline(P, "hermite", 0.0, "centripetal")
    t_fine, _ = trajectory.sample_parameters(sp.t_nodes, 20000)
    k_ref = sp.curvature(t_fine).max()
    assert res.max_curvature >= k_ref * (1 - 1e-6)
    assert res.max_curvature <= k_ref * (1 + 1e-3)
    assert res.max_curvature > res.curvature.max()  # el pico real cae entre muestras

"""Fase 5 — Estudio de convergencia (orden observado en log-log).

Ejecutar con ``pytest tests/test_convergence.py -s`` para ver la tabla de errores.
"""

import numpy as np
import pytest

from core import convergence
from core.convergence import CIRCLE_ARC, LEMNISCATE

CURVES = [CIRCLE_ARC, LEMNISCATE]


@pytest.fixture(scope="module", params=CURVES, ids=lambda c: c.name.split()[0])
def study(request):
    curve = request.param
    res = convergence.convergence_study(curve)
    print(f"\n=== {curve.name} ===")
    print(f"{'metodo':<15}" + "".join(f"{'n=' + str(n):>11}" for n in res['hermite_exact']['n'])
          + f"{'orden':>8}{'orden int.':>12}{'ajuste':>8}")
    for method, r in res.items():
        print(f"{method:<15}" + "".join(f"{e:>11.2e}" for e in r["error"])
              + f"{r['order']:>8.2f}{r['order_interior']:>12.2f}{r['order_fit']:>8.2f}")
    return res


def test_derivada_lemniscata():
    """Verifica la derivada analítica de la lemniscata con diferencias centradas."""
    theta = np.linspace(0, 2 * np.pi, 50)
    h = 1e-6
    fd = (LEMNISCATE.position(theta + h) - LEMNISCATE.position(theta - h)) / (2 * h)
    assert np.allclose(LEMNISCATE.derivative(theta), fd, atol=1e-8)


def test_orden_observado_sintetico():
    n = np.array([4, 8, 16, 32])
    e = 3.0 * (1.0 / n) ** 4
    assert np.isclose(convergence.observed_order(n, e), 4.0)
    assert np.allclose(convergence.local_orders(n, e), 4.0)


def test_hermite_tangentes_exactas_orden_4(study):
    assert abs(study["hermite_exact"]["order"] - 4) < 0.3


def test_sujeto_pendientes_exactas_orden_4(study):
    """Teoría: O(h⁴). En la lemniscata se acerca a 4 desde arriba (4.35 con n=64, 4.09 con n=128)."""
    assert study["clamped_exact"]["order"] > 3.7


def test_frontera_reduce_el_orden_global(study):
    """Catmull-Rom y natural: el interior converge más rápido que el error global (efecto de frontera)."""
    for method in ("catmull_rom", "natural"):
        assert study[method]["order_interior"] > study[method]["order"] + 0.5


def test_metodos_estimados_convergen(study):
    """No se fuerza un valor: solo que el error disminuye; los órdenes se reportan arriba."""
    for method in ("catmull_rom", "natural"):
        err = study[method]["error"]
        assert np.all(np.diff(err) < 0)
        assert study[method]["order"] > 1.5

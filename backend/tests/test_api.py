"""Fase 6 — API con TestClient."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID = {
    "waypoints": [[0, 0], [2, 3], [5, 1], [7, 4]],
    "method": "hermite",
    "tension": 0.0,
    "parametrization": "centripetal",
    "samples_per_segment": 50,
    "limits": {"v_cruise": 5.0, "a_lat_max": 2.0, "a_long_max": 1.5},
}
ARRAY_FIELDS = ["points", "t", "s", "speed", "acceleration", "curvature", "v_max", "v_profile", "time"]


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("method", ["hermite", "natural", "clamped"])
@pytest.mark.parametrize("param", ["uniform", "chord", "centripetal"])
def test_trayectoria_valida(method, param):
    r = client.post("/api/trajectory", json={**VALID, "method": method, "parametrization": param})
    assert r.status_code == 200, r.text
    data = r.json()
    N = 3 * 50 + 1
    for field in ARRAY_FIELDS:
        assert len(data[field]) == N, field
    assert len(data["tangents"]) == 4
    assert data["knots_index"] == [0, 50, 100, 150]
    for k, idx in enumerate(data["knots_index"]):
        assert data["points"][idx] == pytest.approx(VALID["waypoints"][k], abs=1e-9)
    assert set(data["stats"]) == {"length", "time_estimate", "max_curvature"}
    assert data["stats"]["length"] == pytest.approx(data["s"][-1])


def test_valores_por_defecto():
    r = client.post("/api/trajectory", json={"waypoints": [[0, 0], [1, 1]]})
    assert r.status_code == 200


def test_tension_1_es_valida_y_finita():
    r = client.post("/api/trajectory", json={**VALID, "tension": 1.0, "parametrization": "uniform"})
    assert r.status_code == 200


def test_clamped_con_tangentes():
    r = client.post("/api/trajectory", json={**VALID, "method": "clamped", "end_tangents": [[1, 0], [0, 1]]})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "patch, fragment",
    [
        ({"waypoints": [[0, 0]]}, "at least 2"),
        ({"waypoints": [[0, 0], [1, 1], [1, 1], [2, 0]]}, "duplicados"),
        ({"tension": 1.5}, "less than or equal to 1"),
        ({"tension": -0.1}, "greater than or equal to 0"),
        ({"method": "bezier"}, "hermite"),
        ({"parametrization": "foo"}, "centripetal"),
        ({"samples_per_segment": 1}, "greater than or equal to 2"),
        ({"limits": {"v_cruise": 0, "a_lat_max": 2, "a_long_max": 1}}, "greater than 0"),
    ],
)
def test_solicitudes_invalidas(patch, fragment):
    r = client.post("/api/trajectory", json={**VALID, **patch})
    assert r.status_code == 422
    assert fragment in r.text


def test_mensaje_duplicado_claro():
    r = client.post("/api/trajectory", json={**VALID, "waypoints": [[0, 0], [1, 1], [1, 1]]})
    msg = r.json()["detail"][0]["msg"]
    assert "Los waypoints 1 y 2 están duplicados" in msg


def test_cors():
    r = client.options(
        "/api/trajectory",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"

"""Modelos Pydantic v2 del contrato de la API (sección 4 de PLAN.md)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Point = tuple[float, float]

MAX_WAYPOINTS = 200
MAX_TOTAL_SAMPLES = 20_000


class Limits(BaseModel):
    """Límites físicos del dron."""

    model_config = ConfigDict(allow_inf_nan=False)

    v_cruise: float = Field(5.0, gt=0, le=100, description="Velocidad de crucero [m/s]")
    a_lat_max: float = Field(2.0, gt=0, le=100, description="Aceleración lateral máxima [m/s²]")
    a_long_max: float = Field(1.5, gt=0, le=100, description="Aceleración longitudinal máxima [m/s²]")


class TrajectoryRequest(BaseModel):
    """Solicitud de cálculo de trayectoria."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "waypoints": [[0, 0], [2, 3], [5, 1], [7, 4]],
                "method": "hermite",
                "tension": 0.0,
                "parametrization": "centripetal",
                "samples_per_segment": 50,
                "limits": {"v_cruise": 5.0, "a_lat_max": 2.0, "a_long_max": 1.5},
            }
        },
    )

    waypoints: list[Point] = Field(..., min_length=2, max_length=MAX_WAYPOINTS, description="Waypoints [[x, y], ...] en metros")
    method: Literal["hermite", "natural", "clamped"] = "hermite"
    tension: float = Field(0.0, ge=0.0, le=1.0, description="Tensión del spline cardinal (solo hermite)")
    parametrization: Literal["uniform", "chord", "centripetal"] = "centripetal"
    samples_per_segment: int = Field(50, ge=2, le=500)
    limits: Limits = Field(default_factory=Limits)
    end_tangents: tuple[Point, Point] | None = Field(
        None,
        description="Solo clamped: [m_0, m_n] respecto del parámetro t. Si se omite se usan diferencias unilaterales.",
    )

    @model_validator(mode="after")
    def _check_waypoints(self) -> "TrajectoryRequest":
        P = np.asarray(self.waypoints, dtype=float)
        dist = np.linalg.norm(np.diff(P, axis=0), axis=1)
        dup = np.flatnonzero(dist <= 1e-9)
        if dup.size:
            k = int(dup[0])
            raise ValueError(
                f"Los waypoints {k} y {k + 1} están duplicados ({self.waypoints[k]}). "
                "Elimine uno de ellos: dos puntos consecutivos iguales no definen un tramo."
            )
        total = (len(self.waypoints) - 1) * self.samples_per_segment
        if total > MAX_TOTAL_SAMPLES:
            raise ValueError(
                f"Demasiadas muestras ({total}); reduzca samples_per_segment o los waypoints "
                f"(máximo {MAX_TOTAL_SAMPLES})."
            )
        return self


class Stats(BaseModel):
    length: float = Field(..., description="Longitud total [m]")
    time_estimate: float = Field(..., description="Tiempo estimado de recorrido [s]")
    max_curvature: float = Field(..., description="Curvatura máxima [1/m]")


class TrajectoryResponse(BaseModel):
    """Trayectoria muestreada y su análisis. Todos los arreglos tienen la misma longitud N."""

    points: list[Point]
    t: list[float]
    s: list[float] = Field(..., description="Longitud de arco acumulada [m]")
    speed: list[float] = Field(..., description="Rapidez paramétrica ‖p'(t)‖")
    acceleration: list[float] = Field(..., description="Aceleración paramétrica ‖p''(t)‖")
    curvature: list[float] = Field(..., description="Curvatura κ [1/m]")
    v_max: list[float] = Field(..., description="Límite de velocidad por curvatura [m/s]")
    v_profile: list[float] = Field(..., description="Perfil de velocidad factible [m/s]")
    time: list[float] = Field(..., description="Tiempo acumulado [s]")
    tangents: list[Point] = Field(..., description="Tangentes en los nodos")
    knots_index: list[int] = Field(..., description="Índices de `points` que son waypoints")
    stats: Stats


class Health(BaseModel):
    status: Literal["ok"] = "ok"

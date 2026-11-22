"""API FastAPI del simulador de trayectorias.

Ejecutar desde ``backend/``::

    uvicorn api.main:app --reload        # docs en http://localhost:8000/docs

CORS: la variable de entorno ``ALLOWED_ORIGINS`` (lista separada por comas)
define los orígenes permitidos. Por defecto, ``http://localhost:3000``.
"""

from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.trajectory import compute_trajectory

from .schemas import Health, Stats, TrajectoryRequest, TrajectoryResponse

app = FastAPI(
    title="Drone Splines API",
    description="Planificación de trayectorias para drones con splines de Hermite cúbicos y spline cúbico natural.",
    version="1.0.0",
)

_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=Health)
def health() -> Health:
    """Verificación de vida del servicio."""
    return Health()


@app.post("/api/trajectory", response_model=TrajectoryResponse)
def trajectory(req: TrajectoryRequest) -> TrajectoryResponse:
    """Interpola los waypoints y devuelve la trayectoria con su análisis cinemático."""
    try:
        res = compute_trajectory(
            np.asarray(req.waypoints, dtype=float),
            method=req.method,
            tension=req.tension,
            parametrization=req.parametrization,
            samples_per_segment=req.samples_per_segment,
            v_cruise=req.limits.v_cruise,
            a_lat_max=req.limits.a_lat_max,
            a_long_max=req.limits.a_long_max,
            end_tangents=None if req.end_tangents is None else np.asarray(req.end_tangents, dtype=float),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    arrays = (res.points, res.s, res.speed, res.acceleration, res.curvature, res.v_profile, res.time, res.tangents)
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise HTTPException(status_code=422, detail="La trayectoria produjo valores no finitos; revise los waypoints.")

    return TrajectoryResponse(
        points=res.points.tolist(),
        t=res.t.tolist(),
        s=res.s.tolist(),
        speed=res.speed.tolist(),
        acceleration=res.acceleration.tolist(),
        curvature=res.curvature.tolist(),
        v_max=res.v_max.tolist(),
        v_profile=res.v_profile.tolist(),
        time=res.time.tolist(),
        tangents=res.tangents.tolist(),
        knots_index=res.knots_index.tolist(),
        stats=Stats(length=res.length, time_estimate=res.time_estimate, max_curvature=res.max_curvature),
    )

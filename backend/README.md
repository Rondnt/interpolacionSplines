# Backend: núcleo numérico + API

## Módulos de `core/`

Todos los módulos son funciones puras con NumPy, sin dependencias de FastAPI.

| Módulo | Contenido |
|---|---|
| `parametrization.py` | `parametrize(P, mode)`: nodos uniformes, por cuerda o centrípetos. |
| `tangents.py` | `cardinal_tangents(t_nodes, P, tension, scheme)`: Catmull-Rom no uniforme (`parabolic`) o clásica (`central`). `exact_tangents`. |
| `hermite.py` | Bases de Hermite y sus derivadas, `evaluate(t_nodes, P, m, t_eval)` → (p, p′, p″), `second_derivative_jumps`. |
| `cubic_natural.py` | `thomas_solve`, `natural_second_derivatives`, `clamped_second_derivatives`, `evaluate`, `nodal_slopes`. |
| `kinematics.py` | `speed`, `acceleration_norm`, `curvature` (acotada, nunca NaN), `curvature_peaks` (sección áurea), `physical_accelerations`. |
| `arc_length.py` | `simpson_composite`, `cumulative_arc_length`, `reparametrize_by_arc_length` (Newton + bisección). |
| `velocity_profile.py` | `curvature_speed_limit`, `longitudinal_limit` (vectorizado con mínimo acumulado), `cumulative_time`. |
| `trajectory.py` | `compute_trajectory(...)`: encadena todos los pasos (lo usan la API y el notebook). |
| `convergence.py` | Curvas exactas (arco de círculo, lemniscata) y `convergence_study`. |

Notación: `t_nodes` son los nodos, `P` los waypoints, `m` las tangentes (Hermite), `M` las segundas derivadas (spline cúbico), `delta` = Δₖ, `u` el parámetro local y `kappa` = κ.

## API

`POST /api/trajectory`

```json
{
  "waypoints": [[0, 0], [2, 3], [5, 1], [7, 4]],
  "method": "hermite",
  "tension": 0.0,
  "parametrization": "centripetal",
  "samples_per_segment": 50,
  "limits": { "v_cruise": 5.0, "a_lat_max": 2.0, "a_long_max": 1.5 },
  "end_tangents": null
}
```

Valores válidos:

- `method`: `hermite`, `natural` o `clamped`.
- `tension`: entre 0 y 1; solo afecta a `hermite`.
- `parametrization`: `uniform`, `chord` o `centripetal`.
- `samples_per_segment`: entre 2 y 500.
- `end_tangents`: opcional y solo para `clamped`. Si se omite, se usan diferencias unilaterales.

La respuesta contiene arreglos de longitud N = n·samples_per_segment + 1: `points`, `t`, `s`, `speed`, `acceleration`, `curvature`, `v_max`, `v_profile` y `time`. Además trae `tangents` (n+1), `knots_index` (n+1) y `stats` (`length`, `time_estimate`, `max_curvature`).

Se responde con **422 y un mensaje claro** en estos casos:

- menos de 2 waypoints;
- waypoints consecutivos duplicados;
- `tension` fuera de [0, 1];
- límites físicos ≤ 0;
- más de 20 000 muestras en total.

`GET /api/health` → `{"status": "ok"}`

**CORS:** la variable de entorno `ALLOWED_ORIGINS` recibe una lista separada por comas. Por defecto vale `http://localhost:3000`.

## Tests

```bash
pytest -v                               # 96 tests
pytest tests/test_convergence.py -s     # imprime la tabla de errores y órdenes
```

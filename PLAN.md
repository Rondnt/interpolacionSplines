# PLAN DE DESARROLLO — Planificación de Trayectorias para Drones con Splines

**Curso:** Métodos Numéricos — Ingeniería de Software
**Tema:** Interpolación segmentaria (Splines de Hermite cúbicos)
**Problema real:** Un dron/AGV debe recorrer una lista de waypoints sin frenar en cada esquina. Se interpola una trayectoria paramétrica suave (x(t), y(t)) y se analizan velocidad, aceleración y curvatura para obtener un perfil de velocidad físicamente factible.

---

## 0. Reglas obligatorias del proyecto

1. **Los splines se implementan a mano** en `backend/core/` con NumPy. Está **prohibido** usar `scipy.interpolate` (u otra librería de interpolación) dentro de `core/`. SciPy solo se permite en `backend/tests/` para validar resultados.
2. Todo el código numérico va **documentado**: docstrings en español con la fórmula matemática implementada, parámetros, retorno y complejidad.
3. Las funciones de `core/` son **puras**: sin I/O, sin estado global y sin dependencias de FastAPI. Reciben y devuelven arrays de NumPy.
4. El mismo `core/` lo usan la API y el notebook del informe.
5. Commits pequeños por fase con mensajes descriptivos (`feat(core): bases de Hermite`, etc.).

---

## 1. Stack tecnológico

| Capa | Tecnología |
|---|---|
| Núcleo numérico | Python 3.11+, NumPy |
| Backend | FastAPI, Pydantic v2, Uvicorn |
| Tests | pytest, SciPy (solo para validar) |
| Frontend | Next.js (App Router), TypeScript, Tailwind CSS |
| Gráficas | Recharts |
| Dibujo y animación | HTML Canvas nativo + `requestAnimationFrame` |
| Análisis para el informe | Jupyter Notebook + Matplotlib |
| Deploy | Frontend en Vercel, backend en Render |

---

## 2. Estructura del monorepo

```
interpolacionSplines/          # raíz del monorepo (repositorio git)
├── render.yaml                # blueprint de Render (backend)
├── backend/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── parametrization.py   # Parámetro uniforme / cuerda / centrípeto
│   │   ├── tangents.py          # Tangentes Cardinal / Catmull-Rom / exactas
│   │   ├── hermite.py           # Bases de Hermite y evaluación por tramos (C¹)
│   │   ├── cubic_natural.py     # Spline cúbico natural + algoritmo de Thomas (C²)
│   │   ├── kinematics.py        # Derivadas, rapidez, aceleración, curvatura
│   │   ├── arc_length.py        # Longitud de arco (Simpson) + reparametrización
│   │   ├── velocity_profile.py  # v_max por curvatura (+ límite longitudinal)
│   │   ├── trajectory.py        # Orquestación waypoints → trayectoria (API y notebook)
│   │   └── convergence.py       # Curvas exactas y estudio de convergencia
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # App FastAPI, CORS, routers
│   │   └── schemas.py           # Modelos Pydantic
│   ├── tests/
│   │   ├── test_hermite.py
│   │   ├── test_cubic_natural.py
│   │   ├── test_kinematics.py
│   │   ├── test_arc_length.py
│   │   ├── test_velocity_profile.py
│   │   ├── test_convergence.py
│   │   └── test_api.py
│   ├── requirements.txt         # runtime (API)
│   ├── requirements-dev.txt     # tests + notebook
│   ├── pytest.ini
│   └── README.md
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   └── layout.tsx
│   ├── components/
│   │   ├── TrajectoryCanvas.tsx # Plano, clics, arrastre, curva, dron animado
│   │   ├── ControlPanel.tsx     # Método, tensión, parametrización, límites físicos
│   │   ├── ProfileCharts.tsx    # Rapidez, aceleración, curvatura, v_max vs distancia
│   │   └── StatsPanel.tsx       # Longitud total, tiempo estimado, κ máx
│   ├── lib/
│   │   ├── api.ts               # Cliente HTTP tipado
│   │   ├── types.ts             # Tipos que reflejan los schemas de Pydantic
│   │   ├── useTrajectory.ts     # Throttle + descarte de respuestas viejas
│   │   ├── useDroneAnimation.ts # Animación a velocidad física
│   │   └── numeric.ts           # Interpolación lineal, escala de color
│   └── .env.local.example       # NEXT_PUBLIC_API_URL=http://localhost:8000
├── notebooks/
│   └── analisis.ipynb           # Gráficas y tablas para el informe
├── docs/
│   └── figuras/                 # PNG exportados desde el notebook
└── README.md
```

---

## 3. Modelo matemático (referencia para implementar)

### 3.1 Trayectoria paramétrica
Waypoints `P_k = (x_k, y_k)`, `k = 0..n`. Cada coordenada se interpola por separado contra un parámetro `t_k` creciente.

**Parametrización** (`parametrization.py`):
- Uniforme: `t_k = k`
- Cuerda: `t_0 = 0`, `t_{k+1} = t_k + ‖P_{k+1} − P_k‖`
- Centrípeta: `t_{k+1} = t_k + ‖P_{k+1} − P_k‖^{0.5}` (evita bucles y cúspides)

### 3.2 Spline de Hermite cúbico (C¹)
Tramo `[t_k, t_{k+1}]`, `Δ_k = t_{k+1} − t_k`, `u = (t − t_k)/Δ_k ∈ [0,1]`:

```
p(u) = h00(u)·P_k + h10(u)·Δ_k·m_k + h01(u)·P_{k+1} + h11(u)·Δ_k·m_{k+1}

h00 = 2u³ − 3u² + 1        h00' = 6u² − 6u       h00'' = 12u − 6
h10 = u³ − 2u² + u         h10' = 3u² − 4u + 1   h10'' = 6u − 4
h01 = −2u³ + 3u²           h01' = −6u² + 6u      h01'' = −12u + 6
h11 = u³ − u²              h11' = 3u² − 2u       h11'' = 6u − 2
```
Derivadas respecto de t: `dp/dt = (1/Δ_k)·dp/du`, `d²p/dt² = (1/Δ_k²)·d²p/du²`.

### 3.3 Estimación de tangentes (`tangents.py`)
- **Cardinal** con tensión `c ∈ [0,1]` (c = 0 ⇒ Catmull-Rom). Con `d_k = (P_{k+1} − P_k)/Δ_k`:
  `m_k = (1 − c)·(Δ_k·d_{k−1} + Δ_{k−1}·d_k) / (Δ_{k−1} + Δ_k)` (Catmull-Rom no uniforme, Barry-Goldman).
  Con parametrización uniforme coincide con la fórmula clásica `(1 − c)·(P_{k+1} − P_{k−1}) / (t_{k+1} − t_{k−1})`,
  que sigue disponible como `scheme="central"`. La propiedad "la centrípeta no forma bucles ni cúspides" vale para esta tangente.
- Extremos: diferencias unilaterales `m_0 = (P_1 − P_0)/Δ_0`, `m_n = (P_n − P_{n−1})/Δ_{n−1}` (escaladas por `1 − c`).
- **Exactas:** se reciben como entrada (para validación contra curvas analíticas).

### 3.4 Spline cúbico natural (C²) — comparación
Segundas derivadas `M_k` en los nodos, con `M_0 = M_n = 0`. Para `k = 1..n−1`:
```
Δ_{k−1}·M_{k−1} + 2(Δ_{k−1} + Δ_k)·M_k + Δ_k·M_{k+1}
    = 6·[ (y_{k+1} − y_k)/Δ_k − (y_k − y_{k−1})/Δ_{k−1} ]
```
Sistema tridiagonal resuelto con el **algoritmo de Thomas** implementado a mano, O(n). Opción adicional: spline **sujeto** (clamped) con pendientes dadas en los extremos.

### 3.5 Cinemática (`kinematics.py`)
- Rapidez paramétrica: `‖p'(t)‖` (derivada respecto de t, **no** del tiempo físico)
- Curvatura: `κ = |x'·y'' − y'·x''| / (x'² + y'²)^{3/2}`. Si `‖p'‖ ≈ 0` (por ejemplo, nodos con c = 1) es una esquina: se devuelve `κ_max = 10⁶`, nunca NaN.

### 3.6 Longitud de arco (`arc_length.py`)
- `s(t) = ∫ ‖p'(τ)‖ dτ` con **Simpson compuesto** por tramo.
- Reparametrización por longitud de arco: dada una distancia `s*`, hallar `t` tal que `s(t) = s*` (Newton con respaldo de bisección). Se usa para animar el dron a **velocidad física real** y no a t uniforme.

### 3.7 Perfil de velocidad (`velocity_profile.py`)
- Límite por curvatura: `v_max(s) = min( v_crucero, √(a_lat_max / κ(s)) )`
- (Extra) límite longitudinal: pasada hacia adelante y hacia atrás con `a_long_max` para que el dron acelere y frene de forma factible: `v_i² ≤ v_{i−1}² + 2·a_long·Δs`. El dron **parte y llega en reposo**.
- Tiempo total estimado: `T = Σ 2·Δs_i / (v_{i−1} + v_i)` (exacto con aceleración constante entre muestras; `Δs_i / v_i` diverge con v = 0).

---

## 4. Contrato de la API

### `POST /api/trajectory`
**Request**
```json
{
  "waypoints": [[0, 0], [2, 3], [5, 1], [7, 4]],
  "method": "hermite",                 // "hermite" | "natural" | "clamped"
  "tension": 0.0,                      // solo hermite, 0..1
  "parametrization": "centripetal",    // "uniform" | "chord" | "centripetal"
  "samples_per_segment": 50,
  "limits": { "v_cruise": 5.0, "a_lat_max": 2.0, "a_long_max": 1.5 },
  "end_tangents": null                 // opcional, solo clamped: [[m0x, m0y], [mnx, mny]]
}
```
**Response**
```json
{
  "points":      [[x, y], ...],
  "t":           [...],
  "s":           [...],          // longitud de arco acumulada
  "speed":       [...],          // ‖p'(t)‖
  "acceleration":[...],          // ‖p''(t)‖
  "curvature":   [...],
  "v_max":       [...],          // límite por curvatura
  "v_profile":   [...],          // perfil de velocidad factible
  "time":        [...],          // tiempo acumulado (para animar el dron)
  "tangents":    [[mx, my], ...],// tangentes en los nodos
  "knots_index": [...],          // índices de `points` que son waypoints
  "stats": { "length": 0.0, "time_estimate": 0.0, "max_curvature": 0.0 }
}
```
Validaciones: mínimo 2 waypoints, sin waypoints consecutivos duplicados (error 422 con un mensaje claro), `tension` en [0,1], límites físicos > 0, `samples_per_segment` en [2, 500] y como máximo 20 000 muestras.

Muestreo: `u ∈ [0, 1)` en cada tramo más el último nodo ⇒ `N = n·samples_per_segment + 1` y `knots_index[k] = k·samples_per_segment`. En un nodo interior, `p''` es el límite por la derecha.

### `GET /api/health` → `{ "status": "ok" }`

Las docs automáticas quedan disponibles en `/docs`.

---

## 5. Fases de desarrollo

> Cada fase termina con sus tests en verde y un commit. No avanzar a la siguiente sin cumplir los criterios de aceptación.

### Fase 1 — Setup del monorepo
- [x] Estructura de carpetas de la sección 2.
- [x] `backend/requirements.txt` (fastapi, uvicorn[standard], pydantic, numpy) y `requirements-dev.txt` (+ pytest, httpx, scipy, matplotlib, jupyter).
- [x] Next.js con TypeScript + Tailwind en `frontend/`.
- [x] `.gitignore` (venv, node_modules, .next, __pycache__, .ipynb_checkpoints).
- [x] README raíz con instrucciones para correr el proyecto.

**Aceptación:** `uvicorn api.main:app --reload` responde en `/api/health`; `npm run dev` muestra la página.

### Fase 2 — Núcleo Hermite
- [x] `parametrization.py` (3 modos).
- [x] `tangents.py` (cardinal/Catmull-Rom y exactas).
- [x] `hermite.py`: `evaluate(t_nodes, P, M, t_eval)` devuelve posición, primera y segunda derivada (vectorizado).

**Aceptación (tests):**
- La curva pasa exactamente por los waypoints (tolerancia 1e-12).
- Coincide con `scipy.interpolate.CubicHermiteSpline` usando las mismas tangentes (tolerancia 1e-10).
- La primera derivada es continua en los nodos; la segunda **no** tiene por qué serlo (test que mide y reporta el salto).

### Fase 3 — Spline cúbico natural
- [x] `thomas_solve(a, b, c, d)` propio.
- [x] `cubic_natural.py` natural y sujeto, con evaluación de posición y dos derivadas.

**Aceptación:** coincide con `scipy.interpolate.CubicSpline(bc_type='natural'/'clamped')`; continuidad C² verificada numéricamente; Thomas coincide con `numpy.linalg.solve`.

### Fase 4 — Cinemática, longitud de arco y perfil de velocidad
- [x] `kinematics.py`, `arc_length.py`, `velocity_profile.py`.

**Aceptación:**
- Círculo de radio R: curvatura ≈ 1/R y longitud ≈ 2πR (error relativo < 1e-3 con suficientes puntos).
- Simpson compuesto coincide con `scipy.integrate.quad` en un tramo.
- La reparametrización produce puntos aproximadamente equiespaciados en distancia.
- `v_profile ≤ v_cruise` y `v_profile ≤ √(a_lat/κ)` en todos los puntos.

### Fase 5 — Estudio de convergencia (para el informe)
- [x] `test_convergence.py`: curva exacta (arco de círculo / lemniscata de Bernoulli), se muestrean n waypoints con n = 4, 8, 16, 32, 64; se mide el error máximo de posición contra la curva exacta.
- [x] Se calcula el orden observado como la pendiente en log-log.

**Aceptación:** Hermite con tangentes exactas muestra orden ≈ 4 (orden asintótico: último par de n; con n = 4 la lemniscata aún no está resuelta). Se reportan (sin forzar un valor) los órdenes de Hermite con tangentes estimadas y del cúbico natural, documentando el efecto de las condiciones de frontera.

### Fase 6 — API
- [x] `schemas.py` con Pydantic según la sección 4.
- [x] `main.py`: endpoint `/api/trajectory` que orquesta core/; CORS configurable por variable de entorno `ALLOWED_ORIGINS`.
- [x] Tests de la API con `TestClient`.

**Aceptación:** request válido → 200 con todos los campos; inválido → 422 con mensaje claro.

### Fase 7 — Frontend: plano interactivo
- [x] `TrajectoryCanvas`: plano cartesiano con grilla y ejes en metros; clic para agregar waypoint, arrastre para moverlo, clic derecho para borrarlo; botón "Limpiar".
- [x] Al cambiar waypoints o parámetros se llama a la API (debounce de unos 150 ms) y se redibuja la curva.
- [x] Se dibujan los waypoints numerados y las tangentes en los nodos (toggle).
- [x] Coloreado de la curva por velocidad (`v_profile`), con leyenda.
- [x] `ControlPanel`: método, tensión (slider), parametrización y límites físicos.

**Aceptación:** colocar 5 puntos genera la curva al instante; mover un punto actualiza la curva de forma fluida.

### Fase 8 — Animación del dron
- [x] Ícono de dron (SVG simple) que recorre la curva a velocidad física, rotado según la tangente. Se interpola `s(time)` con el arreglo `time` del backend: con "distancia por frame = v·dt" el dron nunca arrancaría, porque parte con v = 0.
- [x] Botones Play / Pausa / Reiniciar y multiplicador de velocidad de simulación.
- [x] Marcador sincronizado en las gráficas en la posición actual del dron.

**Aceptación:** el dron frena visiblemente en las curvas cerradas y acelera en las rectas.

### Fase 9 — Gráficas y modo comparación
- [x] `ProfileCharts`: rapidez, aceleración, curvatura y v_max vs distancia recorrida, con los waypoints marcados como líneas verticales.
- [x] Modo comparación: Hermite vs cúbico natural superpuestos en el canvas y en la gráfica de aceleración (se ve el salto C¹ vs la continuidad C²).
- [x] `StatsPanel`: longitud total, tiempo estimado y curvatura máxima.

### Fase 10 — Notebook y documentación
- [x] `notebooks/analisis.ipynb` importando `backend/core`:
  1. Ejemplo de ruta de almacén con 6 a 8 waypoints.
  2. Bases de Hermite graficadas.
  3. Hermite vs natural: trayectoria, velocidad y aceleración (salto en los nodos).
  4. Efecto de la tensión (c = 0, 0.5, 0.9).
  5. Efecto de la parametrización (uniforme vs cuerda vs centrípeta).
  6. Convergencia log-log con tabla de errores.
  7. Perfil de velocidad y tiempo estimado.
- [x] Exportar figuras a `docs/figuras/`.
- [x] README final: descripción, capturas, instrucciones y explicación del método. Demo: https://interpolacion-splines.vercel.app

### Fase 11 — Deploy
- [x] Backend en Render (`uvicorn api.main:app --host 0.0.0.0 --port $PORT`) con `render.yaml`: https://drone-splines-api.onrender.com
- [x] Frontend en Vercel con `NEXT_PUBLIC_API_URL` (Root Directory: `frontend/`): https://interpolacion-splines.vercel.app
- [x] Configurar CORS con el dominio de Vercel (variable `ALLOWED_ORIGINS` en Render).

---

## 6. Prioridades

- **Mínimo entregable (aprobado):** Fases 1, 2, 4 (solo cinemática), 6, 7, 8 básica y 10.
- **Para destacar:** Fases 3, 5, 9, longitud de arco real y perfil de velocidad con límite longitudinal.

## 7. Mapeo con la rúbrica

| Rúbrica | Evidencia |
|---|---|
| Introducción y contexto | Ruta de almacén / dron de reparto (notebook, sección 1) |
| Modelado matemático | Sección 3 de este plan: variables, parámetros y condiciones (tangentes, fronteras) |
| Resolución analítica y numérica | Curvas exactas (círculo/lemniscata) vs spline; Thomas; Simpson |
| Resultados e interpretación | Gráficas de velocidad/aceleración, convergencia y tiempo estimado |
| Repositorio + código documentado | Monorepo, docstrings, tests |
| App que permite cambiar parámetros | Canvas interactivo + ControlPanel |

## 8. Decisiones y correcciones al plan

Revisión hecha antes de implementar; todas están cubiertas por tests.

| # | Problema en el plan original | Decisión |
|---|---|---|
| 1 | Con `tension = 1` las tangentes son nulas, la rapidez vale 0 en los nodos y κ = 0/0 = NaN; con κ = 0, `√(a_lat/κ) = ∞`. NaN e ∞ no se pueden serializar a JSON. | `curvature` devuelve `KAPPA_MAX = 10⁶` en los puntos singulares (esquinas) y el límite lateral trata κ = 0 como recta. El dron se detiene en las esquinas. |
| 2 | `M` se usaba para las tangentes (`evaluate(t_nodes, P, M, ...)`) y también para las segundas derivadas del spline natural. | `m` = tangentes, `M` = segundas derivadas. |
| 3 | La tangente `(P_{k+1} − P_{k−1})/(t_{k+1} − t_{k−1})` es O(h) con nodos no uniformes, y la propiedad "sin bucles" de la centrípeta no vale con ella. | Tangente de Catmull-Rom no uniforme (parabólica) por defecto; la clásica queda como `scheme="central"`. |
| 4 | El test del círculo con κ ≈ 1/R (< 1e-3) falla con tangentes estimadas: los extremos unilaterales dan un error de ~100 % en κ. | La curvatura se prueba con derivadas analíticas, con Hermite con tangentes exactas (n = 100) y en el interior con Catmull-Rom. |
| 5 | En la convergencia no se fijaba el parámetro de los nodos. | `t_k = θ_k` (parámetro propio de la curva), arco de 3π/2 en vez del círculo cerrado; se reportan los órdenes global, interior y por ajuste. |
| 6 | `T = Σ Δs_i / v_i` diverge si el dron parte del reposo. | `Δt_i = 2Δs_i/(v_{i−1} + v_i)`; para dos paradas seguidas, `2√(Δs/a)`. |
| 7 | Animar con "distancia = v·dt" deja al dron detenido en `v(0) = 0`. | La API devuelve `time` y el frontend interpola `s(time)`. |
| 8 | Las gráficas piden `v_max`, pero la respuesta no lo incluía. | Se agregaron `v_max` y `time` a la respuesta. |
| 9 | El spline sujeto no tenía de dónde sacar las pendientes de los extremos. | `end_tangents` opcional; por defecto, diferencias unilaterales. |
| 10 | Con el debounce puro la curva no se actualiza durante el arrastre, y cancelar solicitudes en vuelo impide que llegue alguna respuesta si el servidor es lento. | Throttle de 150 ms con disparo final y descarte de respuestas viejas por número de secuencia. |
| 11 | `requirements.txt` instalaba Jupyter y SciPy en producción. | Se separó en `requirements.txt` y `requirements-dev.txt`. |
| 12 | La pasada longitudinal es un bucle O(N) en Python. | Vectorizada: `w_i = c_i + mín_{j≤i}(v_max_j² − c_j)` con `np.minimum.accumulate` (validada contra el bucle). |
| 13 | *(Auditoría posterior)* En curvas muy cerradas κ tiene picos más angostos que la malla de muestreo; además, en Hermite κ salta en los nodos y el máximo puede ser el límite por la izquierda. Con 50 muestras por tramo, κ máx se subestimaba hasta en un 99 % y el perfil no frenaba en esos picos. | `kinematics.interval_max_curvature`: máximo de κ en **cada intervalo** entre muestras con sección áurea, conservando el mejor punto visto. Cada muestra usa el peor κ de sus dos intervalos vecinos, y `max_curvature` reporta el pico real. |
| 14 | *(Auditoría posterior)* El límite lateral se cumplía solo en las muestras. Como v se interpola linealmente entre ellas y √(a_lat/κ) es curva, se superaba a_lat entre muestras (0.08 % en la ruta del almacén y hasta 65 % en giros de centímetros). | Con el máximo de κ por intervalo, si v_i y v_{i+1} cumplen v ≤ √(a_lat/κ̂) también lo cumple cualquier interpolación lineal entre ellas: **la garantía vale en toda la curva**. Además, la longitud de arco usa Simpson con 8 subintervalos por muestra (antes 2), porque en esquinas casi exactas el integrando ‖p′‖ cambia muy rápido. Costo: unos 14 ms por cálculo con 8 waypoints. |

**Precisión de los resultados (auditoría con 40 rutas aleatorias × 3 parametrizaciones):**

- Posición, derivadas, tangentes y κ coinciden con SciPy (`CubicHermiteSpline`, `CubicSpline`) con un error relativo de 1e-11 o menos.
- Catmull-Rom coincide con el algoritmo piramidal de Barry-Goldman (error ~1e-16).
- La longitud de arco coincide con `scipy.integrate.quad` (error relativo de 4e-5 o menos).
- La API devuelve exactamente lo que calcula el núcleo.
- Las restricciones físicas se cumplen **en toda la curva, también entre muestras**. En la ruta del almacén el exceso de a_lat medido en una malla 40 veces más fina es 0 (2·10⁻¹⁶). En 360 combinaciones aleatorias (3 métodos × 3 parametrizaciones × tensiones 0/0.5/0.9) el peor caso es 0.15 %: una esquina casi exacta con tensión 0.9, recorrida a 2.5 mm/s en un radio de 3 µm. Ese valor queda al nivel de la precisión de la propia malla de comprobación.
- κ máx coincide con una malla de 20 000 puntos por tramo (o la supera cuando esa malla tampoco ve el pico).

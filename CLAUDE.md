# CLAUDE.md — Instrucciones para Claude Code

Proyecto universitario de Métodos Numéricos: simulador web de trayectorias de drones con **Splines de Hermite cúbicos**. El plan completo está en `PLAN.md`; síguelo fase por fase. La sección 8 de PLAN.md registra las decisiones que corrigen el plan original.

## Reglas
- Trabaja **una fase de PLAN.md a la vez**. Al terminar, corre los tests, marca los checkboxes y propone el mensaje de commit.
- **Nunca** uses `scipy.interpolate` ni otra librería de interpolación dentro de `backend/core/`. SciPy solo se permite en `backend/tests/`.
- Las funciones de `core/` son puras, vectorizadas con NumPy, con type hints y docstrings **en español** que incluyan la fórmula implementada.
- Los nombres de variables siguen la notación del plan: `t_nodes` (nodos), `P` (waypoints), `m` (tangentes de Hermite), `M` (segundas derivadas del spline cúbico), `delta` (Δₖ), `u` (parámetro local) y `kappa` (κ).
- Ningún arreglo que devuelva la API puede contener NaN ni ∞ (por ejemplo, κ se acota con `KAPPA_MAX`).
- Frontend: TypeScript estricto, componentes funcionales, Tailwind para estilos y sin librerías de UI pesadas.
- No inventes requisitos fuera del plan; si algo es ambiguo, pregunta.

## Comandos
```bash
# Backend
cd backend
python -m venv .venv
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # requirements.txt = solo runtime (Render)
uvicorn api.main:app --reload         # http://localhost:8000/docs
pytest -v                             # pytest tests/test_convergence.py -s para la tabla de órdenes

# Frontend
cd frontend
npm install
npm run dev                           # http://localhost:3000
npm run lint && npm run build

# Notebook (desde notebooks/, con el venv del backend)
python -m jupyter nbconvert --to notebook --execute --inplace analisis.ipynb
```

import type { DroneState, Point, TrajectoryResponse } from "./types";

/** Índice i tal que xs[i] ≤ x < xs[i+1] (búsqueda binaria; xs creciente). */
export function bracket(xs: number[], x: number): number {
  let lo = 0;
  let hi = xs.length - 1;
  if (x <= xs[0]) return 0;
  if (x >= xs[hi]) return Math.max(0, hi - 1);
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  return lo;
}

/** Interpolación lineal de ys(xs) en x. */
export function interp(x: number, xs: number[], ys: number[]): number {
  const i = bracket(xs, x);
  const dx = xs[i + 1] - xs[i];
  const w = dx > 0 ? Math.min(1, Math.max(0, (x - xs[i]) / dx)) : 0;
  return ys[i] + w * (ys[i + 1] - ys[i]);
}

/**
 * Estado del dron en el instante simulado `time`.
 *
 * Se usa el arreglo `time` (tiempo acumulado) que devuelve el backend: s = s(time).
 * Integrar "distancia por frame = v·dt" no sirve aquí porque el dron parte del reposo
 * (v = 0) y nunca arrancaría; el tiempo acumulado ya es la integral exacta de ds/v.
 */
export function droneAt(res: TrajectoryResponse, time: number): DroneState {
  const T = res.time[res.time.length - 1];
  const tt = Math.min(Math.max(time, 0), T);
  const s = interp(tt, res.time, res.s);
  const i = bracket(res.s, s);
  const ds = res.s[i + 1] - res.s[i];
  const w = ds > 0 ? (s - res.s[i]) / ds : 0;
  const [x0, y0] = res.points[i];
  const [x1, y1] = res.points[i + 1];
  // Rumbo: dirección de la cuerda local; si es degenerada se usa la siguiente.
  let dx = x1 - x0;
  let dy = y1 - y0;
  for (let j = i + 1; dx * dx + dy * dy < 1e-18 && j + 1 < res.points.length; j++) {
    dx = res.points[j + 1][0] - res.points[j][0];
    dy = res.points[j + 1][1] - res.points[j][1];
  }
  return {
    x: x0 + w * (x1 - x0),
    y: y0 + w * (y1 - y0),
    heading: Math.atan2(dy, dx),
    s,
    v: interp(s, res.s, res.v_profile),
    time: tt,
  };
}

/** Submuestreo uniforme para gráficas (conserva siempre los índices de `keep`). */
export function decimateIndices(n: number, max: number, keep: number[] = []): number[] {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const step = (n - 1) / (max - 1);
  const set = new Set<number>(keep);
  for (let k = 0; k < max; k++) set.add(Math.round(k * step));
  return Array.from(set).sort((a, b) => a - b);
}

/** Escala de color para la velocidad: rojo (lento, frenando) → amarillo → verde → azul (rápido). */
const STOPS: [number, number, number][] = [
  [220, 38, 38],
  [234, 88, 12],
  [202, 138, 4],
  [22, 163, 74],
  [37, 99, 235],
];

export function speedColor(fraction: number): string {
  const f = Math.min(1, Math.max(0, fraction)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(f));
  const w = f - i;
  const c = STOPS[i].map((v, k) => Math.round(v + w * (STOPS[i + 1][k] - v)));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export const SPEED_GRADIENT_CSS = `linear-gradient(to right, ${STOPS.map((c) => `rgb(${c.join(",")})`).join(", ")})`;

export function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function fmt(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return "—";
  return x.toLocaleString("es", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

"use client";

import { useCallback, useEffect, useRef, useState, type PointerEvent, type RefObject } from "react";
import { SPEED_GRADIENT_CSS, distance, speedColor } from "@/lib/numeric";
import type { DroneState, Point, TrajectoryResponse } from "@/lib/types";

// Plano de trabajo en metros
export const WORLD = { width: 24, height: 14 };
const PAD = { left: 40, right: 14, top: 14, bottom: 34 };
const HIT_RADIUS = 13; // px
const MIN_SEPARATION = 0.05; // m: evita waypoints consecutivos duplicados

const COLORS = {
  bg: "#ffffff",
  gridMinor: "#eef2f7",
  gridMajor: "#d9e1ec",
  axis: "#64748b",
  text: "#475569",
  compare: "#f97316",
  tangent: "#7c3aed",
  waypoint: "#0f172a",
};

interface Size {
  w: number;
  h: number;
  dpr: number;
}

interface Transform {
  scale: number;
  ox: number;
  oy: number;
}

function makeTransform({ w, h }: Size): Transform {
  const availW = w - PAD.left - PAD.right;
  const availH = h - PAD.top - PAD.bottom;
  const scale = Math.max(1, Math.min(availW / WORLD.width, availH / WORLD.height));
  const ox = PAD.left + (availW - scale * WORLD.width) / 2;
  const oy = h - PAD.bottom - (availH - scale * WORLD.height) / 2;
  return { scale, ox, oy };
}

const toScreen = (T: Transform, [x, y]: Point): Point => [T.ox + x * T.scale, T.oy - y * T.scale];
const toWorld = (T: Transform, px: number, py: number): Point => [(px - T.ox) / T.scale, (T.oy - py) / T.scale];

function drawGrid(ctx: CanvasRenderingContext2D, T: Transform) {
  const [x0, y0] = toScreen(T, [0, 0]);
  const [x1, y1] = toScreen(T, [WORLD.width, WORLD.height]);
  ctx.fillStyle = "#fbfcfe";
  ctx.fillRect(x0, y1, x1 - x0, y0 - y1);

  for (let major = 0; major < 2; major++) {
    ctx.beginPath();
    ctx.strokeStyle = major ? COLORS.gridMajor : COLORS.gridMinor;
    ctx.lineWidth = 1;
    for (let x = 0; x <= WORLD.width; x++) {
      if ((x % 2 === 0) !== Boolean(major)) continue;
      const px = Math.round(T.ox + x * T.scale) + 0.5;
      ctx.moveTo(px, y0);
      ctx.lineTo(px, y1);
    }
    for (let y = 0; y <= WORLD.height; y++) {
      if ((y % 2 === 0) !== Boolean(major)) continue;
      const py = Math.round(T.oy - y * T.scale) + 0.5;
      ctx.moveTo(x0, py);
      ctx.lineTo(x1, py);
    }
    ctx.stroke();
  }

  // Ejes
  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.moveTo(x0, y1);
  ctx.lineTo(x0, y0);
  ctx.lineTo(x1, y0);
  ctx.stroke();

  ctx.fillStyle = COLORS.text;
  ctx.font = "11px var(--font-geist-mono), ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let x = 0; x <= WORLD.width; x += 2) ctx.fillText(String(x), T.ox + x * T.scale, y0 + 5);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let y = 0; y <= WORLD.height; y += 2) ctx.fillText(String(y), x0 - 6, T.oy - y * T.scale);

  ctx.font = "11px var(--font-geist-sans), system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText("x [m]", x1, y0 + 19);
  ctx.save();
  ctx.translate(x0 - 30, y1);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "right";
  ctx.fillText("y [m]", 0, 0);
  ctx.restore();
}

function drawPolyline(ctx: CanvasRenderingContext2D, T: Transform, pts: Point[]) {
  ctx.beginPath();
  pts.forEach((p, i) => {
    const [x, y] = toScreen(T, p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawSpeedCurve(ctx: CanvasRenderingContext2D, T: Transform, res: TrajectoryResponse, vCruise: number) {
  // Halo para separar la curva de la grilla
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 7;
  drawPolyline(ctx, T, res.points);

  // Agrupa los segmentos por color (32 niveles) para dibujar pocos trazos
  const LEVELS = 32;
  const buckets: number[][] = Array.from({ length: LEVELS }, () => []);
  for (let i = 0; i < res.points.length - 1; i++) {
    const v = 0.5 * (res.v_profile[i] + res.v_profile[i + 1]);
    const level = Math.min(LEVELS - 1, Math.floor((v / vCruise) * LEVELS));
    buckets[level].push(i);
  }
  ctx.lineWidth = 4;
  buckets.forEach((idx, level) => {
    if (!idx.length) return;
    ctx.strokeStyle = speedColor((level + 0.5) / LEVELS);
    ctx.beginPath();
    for (const i of idx) {
      const [ax, ay] = toScreen(T, res.points[i]);
      const [bx, by] = toScreen(T, res.points[i + 1]);
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();
  });
}

function drawArrow(ctx: CanvasRenderingContext2D, from: Point, to: Point) {
  const [ax, ay] = from;
  const [bx, by] = to;
  const len = Math.hypot(bx - ax, by - ay);
  if (len < 2) return;
  const ang = Math.atan2(by - ay, bx - ax);
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(bx - 8 * Math.cos(ang - 0.4), by - 8 * Math.sin(ang - 0.4));
  ctx.lineTo(bx - 8 * Math.cos(ang + 0.4), by - 8 * Math.sin(ang + 0.4));
  ctx.closePath();
  ctx.fill();
}

/**
 * Tangentes en los nodos. Se dibuja el vector Δ_k·m_k/3: es la distancia del punto de
 * control de Bézier equivalente al tramo de Hermite, así la flecha tiene escala geométrica.
 */
function drawTangents(ctx: CanvasRenderingContext2D, T: Transform, res: TrajectoryResponse) {
  ctx.strokeStyle = COLORS.tangent;
  ctx.fillStyle = COLORS.tangent;
  ctx.lineWidth = 2;
  const K = res.knots_index;
  K.forEach((idx, k) => {
    const delta = k < K.length - 1 ? res.t[K[k + 1]] - res.t[idx] : res.t[idx] - res.t[K[k - 1]];
    const P = res.points[idx];
    const m = res.tangents[k];
    const end: Point = [P[0] + (m[0] * delta) / 3, P[1] + (m[1] * delta) / 3];
    drawArrow(ctx, toScreen(T, P), toScreen(T, end));
  });
}

function drawWaypoints(ctx: CanvasRenderingContext2D, T: Transform, wps: Point[], active: number | null) {
  ctx.font = "600 11px var(--font-geist-sans), system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  wps.forEach((p, k) => {
    const [x, y] = toScreen(T, p);
    const isActive = k === active;
    ctx.beginPath();
    ctx.arc(x, y, isActive ? 10 : 8.5, 0, 2 * Math.PI);
    ctx.fillStyle = isActive ? COLORS.waypoint : "#ffffff";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = COLORS.waypoint;
    ctx.stroke();
    ctx.fillStyle = isActive ? "#ffffff" : COLORS.waypoint;
    ctx.fillText(String(k), x, y + 0.5);
  });
}

// Dron (vista superior) definido con trazos SVG y dibujado con Path2D. Apunta hacia +x.
const circle = (cx: number, cy: number, r: number) =>
  `M${cx - r} ${cy}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
const DRONE_SVG = {
  arms: "M-9 -9L9 9M-9 9L9 -9",
  rotors: [circle(9, -9, 6.5), circle(9, 9, 6.5), circle(-9, -9, 6.5), circle(-9, 9, 6.5)].join(""),
  body: "M-6 -5h9l4 5l-4 5h-9a2 2 0 0 1 -2 -2v-6a2 2 0 0 1 2 -2z",
};
let dronePaths: { arms: Path2D; rotors: Path2D; body: Path2D } | null = null;

function drawDrone(ctx: CanvasRenderingContext2D, T: Transform, d: DroneState) {
  dronePaths ??= {
    arms: new Path2D(DRONE_SVG.arms),
    rotors: new Path2D(DRONE_SVG.rotors),
    body: new Path2D(DRONE_SVG.body),
  };
  const [x, y] = toScreen(T, [d.x, d.y]);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-d.heading); // y de pantalla crece hacia abajo
  ctx.scale(1.15, 1.15);
  ctx.shadowColor = "rgba(15, 23, 42, 0.35)";
  ctx.shadowBlur = 8;
  ctx.shadowOffsetY = 2;
  ctx.fillStyle = "rgba(148, 163, 184, 0.35)";
  ctx.fill(dronePaths.rotors);
  ctx.shadowColor = "transparent";
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 2.5;
  ctx.stroke(dronePaths.arms);
  ctx.lineWidth = 1.5;
  ctx.stroke(dronePaths.rotors);
  ctx.fillStyle = "#0f172a";
  ctx.fill(dronePaths.body);
  ctx.fillStyle = "#f43f5e";
  ctx.beginPath();
  ctx.arc(3.5, 0, 1.8, 0, 2 * Math.PI);
  ctx.fill();
  ctx.restore();
}

interface Props {
  waypoints: Point[];
  onChange: (waypoints: Point[]) => void;
  result: TrajectoryResponse | null;
  compare: TrajectoryResponse | null;
  showTangents: boolean;
  vCruise: number;
  droneRef: RefObject<DroneState | null>;
}

export default function TrajectoryCanvas({
  waypoints,
  onChange,
  result,
  compare,
  showTangents,
  vCruise,
  droneRef,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const baseRef = useRef<HTMLCanvasElement>(null);
  const topRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<Size | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const dragRef = useRef<number | null>(null);
  const wpsRef = useRef(waypoints);
  wpsRef.current = waypoints;

  // Tamaño responsivo con soporte para pantallas de alta densidad
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ w: width, h: height, dpr: window.devicePixelRatio || 1 });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Capa base: grilla, curvas, tangentes y waypoints
  useEffect(() => {
    const canvas = baseRef.current;
    if (!canvas || !size) return;
    canvas.width = Math.round(size.w * size.dpr);
    canvas.height = Math.round(size.h * size.dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, size.w, size.h);
    const T = makeTransform(size);
    drawGrid(ctx, T);

    if (waypoints.length >= 2 && !result) {
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "#cbd5e1";
      ctx.lineWidth = 1.5;
      drawPolyline(ctx, T, waypoints);
      ctx.setLineDash([]);
    }
    if (compare) {
      ctx.setLineDash([7, 5]);
      ctx.strokeStyle = COLORS.compare;
      ctx.lineWidth = 2.25;
      drawPolyline(ctx, T, compare.points);
      ctx.setLineDash([]);
    }
    if (result) drawSpeedCurve(ctx, T, result, vCruise);
    if (result && showTangents) drawTangents(ctx, T, result);
    drawWaypoints(ctx, T, waypoints, active);
  }, [size, waypoints, result, compare, showTangents, vCruise, active]);

  // Capa superior: dron animado (se redibuja en cada frame leyendo droneRef)
  useEffect(() => {
    const canvas = topRef.current;
    if (!canvas || !size) return;
    canvas.width = Math.round(size.w * size.dpr);
    canvas.height = Math.round(size.h * size.dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const T = makeTransform(size);
    let raf = 0;
    const frame = () => {
      ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
      ctx.clearRect(0, 0, size.w, size.h);
      const d = droneRef.current;
      if (d) drawDrone(ctx, T, d);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [size, droneRef]);

  const eventToWorld = useCallback(
    (e: PointerEvent<HTMLCanvasElement>): { px: number; py: number; p: Point } | null => {
      if (!size) return null;
      const rect = e.currentTarget.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const [x, y] = toWorld(makeTransform(size), px, py);
      const clamp = (v: number, max: number) => Math.round(Math.min(max, Math.max(0, v)) * 100) / 100;
      return { px, py, p: [clamp(x, WORLD.width), clamp(y, WORLD.height)] };
    },
    [size],
  );

  const hitTest = useCallback(
    (px: number, py: number): number | null => {
      if (!size) return null;
      const T = makeTransform(size);
      let best: number | null = null;
      let bestD = HIT_RADIUS;
      wpsRef.current.forEach((p, k) => {
        const [x, y] = toScreen(T, p);
        const d = Math.hypot(x - px, y - py);
        if (d <= bestD) {
          bestD = d;
          best = k;
        }
      });
      return best;
    },
    [size],
  );

  /** Mueve el waypoint k sin dejarlo encima de un vecino (duplicado consecutivo). */
  const moveTo = useCallback(
    (k: number, p: Point) => {
      const wps = wpsRef.current;
      const prev = wps[k - 1];
      const next = wps[k + 1];
      if ((prev && distance(prev, p) < MIN_SEPARATION) || (next && distance(next, p) < MIN_SEPARATION)) return;
      const copy = wps.slice();
      copy[k] = p;
      onChange(copy);
    },
    [onChange],
  );

  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const ev = eventToWorld(e);
    if (!ev) return;
    const hit = hitTest(ev.px, ev.py);
    e.currentTarget.setPointerCapture(e.pointerId);
    if (hit !== null) {
      dragRef.current = hit;
      setActive(hit);
      return;
    }
    const wps = wpsRef.current;
    const last = wps[wps.length - 1];
    if (last && distance(last, ev.p) < MIN_SEPARATION) return;
    onChange([...wps, ev.p]);
    dragRef.current = wps.length; // se puede arrastrar el punto recién creado
    setActive(wps.length);
  };

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const ev = eventToWorld(e);
    if (!ev) return;
    if (dragRef.current !== null) {
      moveTo(dragRef.current, ev.p);
      return;
    }
    const hit = hitTest(ev.px, ev.py);
    if (hit !== active) setActive(hit);
  };

  const onPointerUp = (e: PointerEvent<HTMLCanvasElement>) => {
    dragRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const onContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (!size) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top);
    if (hit === null) return;
    onChange(wpsRef.current.filter((_, k) => k !== hit));
    setActive(null);
  };

  return (
    <div ref={wrapRef} className="relative aspect-[16/10] w-full select-none overflow-hidden rounded-lg">
      <canvas ref={baseRef} className="absolute inset-0 h-full w-full" />
      <canvas
        ref={topRef}
        className="absolute inset-0 h-full w-full touch-none"
        style={{ cursor: active !== null ? (dragRef.current !== null ? "grabbing" : "grab") : "crosshair" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => dragRef.current === null && setActive(null)}
        onContextMenu={onContextMenu}
        aria-label="Plano de la trayectoria. Clic para agregar un waypoint, arrastrar para moverlo, clic derecho para borrarlo."
        role="img"
      />

      {waypoints.length < 2 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <p className="rounded-md bg-white/90 px-4 py-2 text-sm text-slate-600 shadow-sm ring-1 ring-slate-200">
            Haz clic en el plano para agregar {waypoints.length === 0 ? "waypoints" : "otro waypoint"}
          </p>
        </div>
      )}

    </div>
  );
}

/** Leyenda del plano (fuera del canvas para no tapar la trayectoria). */
export function CanvasLegend({
  vCruise,
  compareLabel,
  showCompare,
  showTangents,
}: {
  vCruise: number;
  compareLabel: string;
  showCompare: boolean;
  showTangents: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-500">
      <span className="flex items-center gap-2">
        Velocidad
        <span className="font-mono tabular-nums">0</span>
        <span className="h-2 w-28 rounded-full" style={{ background: SPEED_GRADIENT_CSS }} />
        <span className="font-mono tabular-nums">{vCruise} m/s</span>
      </span>
      {showCompare && (
        <span className="flex items-center gap-1.5">
          <span className="w-5 border-t-2 border-dashed" style={{ borderColor: COLORS.compare }} />
          {compareLabel}
        </span>
      )}
      {showTangents && (
        <span className="flex items-center gap-1.5">
          <span className="w-5 border-t-2" style={{ borderColor: COLORS.tangent }} />
          Tangente Δₖ·mₖ/3
        </span>
      )}
    </div>
  );
}

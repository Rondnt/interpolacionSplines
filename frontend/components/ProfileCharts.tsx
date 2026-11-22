"use client";

import { memo, useMemo } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { decimateIndices } from "@/lib/numeric";
import type { TrajectoryResponse } from "@/lib/types";

const MAX_POINTS = 600;
const KAPPA_SINGULAR = 1e5;
const PRIMARY = "#0284c7";
const COMPARE = "#f97316";
const LIMIT = "#94a3b8";

type Row = { s: number; y: number };
type Field = "v_profile" | "v_max" | "curvature" | "speed" | "acceleration";

function series(res: TrajectoryResponse, field: Field, cap = Infinity): Row[] {
  const idx = decimateIndices(res.s.length, MAX_POINTS, res.knots_index);
  return idx.map((i) => ({ s: res.s[i], y: Math.min(res[field][i], cap) }));
}

/** Cota para mostrar κ: ignora los valores singulares (esquinas con κ → ∞). */
function kappaCap(res: TrajectoryResponse): number {
  let max = 0;
  for (const k of res.curvature) if (k < KAPPA_SINGULAR && k > max) max = k;
  return max > 0 ? max * 1.15 : 1;
}

interface LineSpec {
  data: Row[];
  color: string;
  name: string;
  dashed?: boolean;
  width?: number;
}

interface ChartProps {
  title: string;
  formula: string;
  unit: string;
  note?: string;
  lines: LineSpec[];
  knots: number[];
  length: number;
  markerS: number | null;
}

const fmtTick = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

const Chart = memo(function Chart({ title, formula, unit, note, lines, knots, length }: Omit<ChartProps, "markerS">) {
  return (
    <div className="rounded-xl bg-white p-3 ring-1 ring-slate-200">
      <div className="mb-1 flex items-baseline justify-between gap-2 px-1">
        <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
        <span className="text-[11px] italic text-slate-400">{formula}</span>
      </div>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 14, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="#f1f5f9" />
            <XAxis
              type="number"
              dataKey="s"
              domain={[0, length]}
              allowDataOverflow
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickFormatter={(v: number) => v.toFixed(0)}
              label={{ value: "s [m]", position: "insideBottomRight", offset: -2, fontSize: 10, fill: "#94a3b8" }}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#64748b" }}
              width={44}
              tickFormatter={fmtTick}
              label={{ value: unit, angle: -90, position: "insideLeft", offset: 12, fontSize: 10, fill: "#94a3b8" }}
            />
            <Tooltip
              formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
              labelFormatter={(v) => `s = ${Number(v).toFixed(2)} m`}
              contentStyle={{ fontSize: 11, borderRadius: 8 }}
            />
            {knots.map((s, k) => (
              <ReferenceLine
                key={k}
                x={s}
                stroke="#e2e8f0"
                strokeDasharray="3 3"
                label={{ value: String(k), position: "top", fontSize: 9, fill: "#94a3b8" }}
              />
            ))}
            {lines.map((l) => (
              <Line
                key={l.name}
                data={l.data}
                dataKey="y"
                name={l.name}
                stroke={l.color}
                strokeWidth={l.width ?? 2}
                strokeDasharray={l.dashed ? "5 4" : undefined}
                dot={false}
                isAnimationActive={false}
                type="linear"
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {note && <p className="mt-1 px-1 text-[11px] leading-snug text-slate-400">{note}</p>}
    </div>
  );
});

/** Marcador del dron como capa HTML encima de la gráfica (evita re-renderizar Recharts en cada frame). */
function ChartWithMarker(props: ChartProps) {
  const { markerS, length } = props;
  // Márgenes coherentes con los del LineChart: izquierda = ancho del eje Y (44) + padding 12; derecha = 12 + 12
  const left = 12 + 44;
  const right = 12 + 12;
  const frac = markerS !== null && length > 0 ? Math.min(1, Math.max(0, markerS / length)) : null;
  return (
    <div className="relative">
      <Chart {...props} />
      {frac !== null && (
        <div
          className="pointer-events-none absolute top-[44px] h-[150px] w-0 border-l-2 border-rose-500/80"
          style={{ left: `calc(${left}px + (100% - ${left + right}px) * ${frac})` }}
        />
      )}
    </div>
  );
}

interface Props {
  result: TrajectoryResponse | null;
  compare: TrajectoryResponse | null;
  primaryLabel: string;
  compareLabel: string;
  markerS: number | null;
}

export default function ProfileCharts({ result, compare, primaryLabel, compareLabel, markerS }: Props) {
  const charts = useMemo(() => {
    if (!result) return null;
    const knots = result.knots_index.map((i) => result.s[i]);
    const length = result.stats.length;
    const cap = Math.max(kappaCap(result), compare ? kappaCap(compare) : 0);
    const cmp = (field: Field, c = Infinity): LineSpec[] =>
      compare ? [{ data: series(compare, field, c), color: COMPARE, name: compareLabel, width: 1.75 }] : [];

    const specs: Omit<ChartProps, "markerS">[] = [
      {
        title: "Velocidad del dron",
        formula: "v(s) ≤ min(v_c, √(a_lat/κ))",
        unit: "m/s",
        note: "Línea punteada: límite por curvatura v_max. El perfil frena antes de las curvas cerradas por el límite longitudinal.",
        lines: [
          { data: series(result, "v_max"), color: LIMIT, name: "v_max", dashed: true, width: 1.5 },
          ...cmp("v_profile"),
          { data: series(result, "v_profile"), color: PRIMARY, name: primaryLabel },
        ],
        knots,
        length,
      },
      {
        title: "Curvatura",
        formula: "κ = |x′y″ − y′x″| / ‖p′‖³",
        unit: "1/m",
        note: "Geométrica: no depende de la parametrización. Los picos marcan dónde el dron debe frenar.",
        lines: [...cmp("curvature", cap), { data: series(result, "curvature", cap), color: PRIMARY, name: primaryLabel }],
        knots,
        length,
      },
      {
        title: "Rapidez paramétrica",
        formula: "‖p′(t)‖",
        unit: "‖dp/dt‖",
        note: "Derivada respecto del parámetro t, no del tiempo físico: sus unidades dependen de la parametrización.",
        lines: [...cmp("speed"), { data: series(result, "speed"), color: PRIMARY, name: primaryLabel }],
        knots,
        length,
      },
      {
        title: "Aceleración paramétrica",
        formula: "‖p″(t)‖",
        unit: "‖d²p/dt²‖",
        note: "Hermite es C¹: p″ salta en los waypoints. El spline cúbico es C²: p″ es continua.",
        lines: [...cmp("acceleration"), { data: series(result, "acceleration"), color: PRIMARY, name: primaryLabel }],
        knots,
        length,
      },
    ];
    return specs;
  }, [result, compare, primaryLabel, compareLabel]);

  if (!charts) {
    return (
      <div className="rounded-xl bg-white p-8 text-center text-sm text-slate-400 ring-1 ring-slate-200">
        Las gráficas aparecerán cuando haya una trayectoria.
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {charts.map((c) => (
        <ChartWithMarker key={c.title} {...c} markerS={markerS} />
      ))}
    </div>
  );
}

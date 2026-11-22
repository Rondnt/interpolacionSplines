"use client";

import { fmt } from "@/lib/numeric";
import type { DroneState, TrajectoryResponse } from "@/lib/types";

interface Props {
  result: TrajectoryResponse | null;
  compare: TrajectoryResponse | null;
  compareLabel: string;
  drone: DroneState | null;
}

const KAPPA_SINGULAR = 1e5; // κ ≥ este valor: esquina (rapidez paramétrica nula)

interface Stat {
  label: string;
  value: string;
  unit: string;
}

function stats(res: TrajectoryResponse): Stat[] {
  const { length, time_estimate, max_curvature } = res.stats;
  const corner = max_curvature >= KAPPA_SINGULAR;
  return [
    { label: "Longitud total", value: fmt(length), unit: "m" },
    { label: "Tiempo estimado", value: fmt(time_estimate), unit: "s" },
    { label: "Velocidad media", value: fmt(time_estimate > 0 ? length / time_estimate : NaN), unit: "m/s" },
    { label: "Curvatura máx.", value: corner ? "esquina" : fmt(max_curvature, 3), unit: corner ? "" : "1/m" },
    { label: "Radio de giro mín.", value: corner ? "0" : fmt(1 / max_curvature), unit: "m" },
  ];
}

export default function StatsPanel({ result, compare, compareLabel, drone }: Props) {
  if (!result) return null;
  const a = stats(result);
  const b = compare ? stats(compare) : null;
  const L = result.stats.length;
  const progress = drone && L > 0 ? (100 * drone.s) / L : 0;

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
      {a.map((stat, i) => (
        <div key={stat.label} className="rounded-lg bg-slate-50 px-3 py-2 ring-1 ring-slate-200/70">
          <div className="text-[11px] text-slate-500">{stat.label}</div>
          <div className="mt-0.5 font-mono text-base tabular-nums text-slate-900">
            {stat.value}
            <span className="ml-1 text-xs text-slate-400">{stat.unit}</span>
          </div>
          {b && (
            <div className="font-mono text-[11px] tabular-nums text-orange-600" title={compareLabel}>
              {b[i].value} {b[i].unit} <span className="font-sans text-orange-400">{compareLabel.toLowerCase()}</span>
            </div>
          )}
        </div>
      ))}

      <div className="rounded-lg bg-slate-900 px-3 py-2 text-white" aria-live="off">
        <div className="flex items-baseline justify-between text-[11px] text-slate-400">
          <span>Dron</span>
          <span className="font-mono tabular-nums">t = {fmt(drone?.time ?? 0, 1)} s</span>
        </div>
        <div className="mt-0.5 font-mono text-base tabular-nums">
          {fmt(drone?.v ?? 0)}
          <span className="ml-1 text-xs text-slate-400">m/s</span>
        </div>
        <div className="mt-1 flex items-center gap-2">
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-slate-700">
            <div className="h-full bg-sky-400" style={{ width: `${progress}%` }} />
          </div>
          <span className="font-mono text-[10px] tabular-nums text-slate-400">{fmt(drone?.s ?? 0, 1)} m</span>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useMemo, useState } from "react";
import ControlPanel from "@/components/ControlPanel";
import ProfileCharts from "@/components/ProfileCharts";
import StatsPanel from "@/components/StatsPanel";
import TrajectoryCanvas, { CanvasLegend } from "@/components/TrajectoryCanvas";
import { useDroneAnimation } from "@/lib/useDroneAnimation";
import { useTrajectory } from "@/lib/useTrajectory";
import { METHOD_LABEL, type Method, type Point, type TrajectoryParams, type TrajectoryRequest } from "@/lib/types";

/** Ruta de ejemplo: un dron de reparto que recorre los pasillos de un almacén. */
const WAREHOUSE_ROUTE: Point[] = [
  [2, 2],
  [9, 2.5],
  [11.5, 5],
  [11.5, 9.5],
  [14, 12],
  [20, 12],
  [22, 8.5],
  [18.5, 5],
];

const DEFAULT_PARAMS: TrajectoryParams = {
  method: "hermite",
  tension: 0,
  parametrization: "centripetal",
  limits: { v_cruise: 5, a_lat_max: 2, a_long_max: 1.5 },
};

const SAMPLES_PER_SEGMENT = 50;
const SIM_SPEEDS = [0.5, 1, 2, 4];

const SHORT_LABEL: Record<Method, string> = { hermite: "Hermite", natural: "Natural", clamped: "Sujeto" };

function IconButton({
  onClick,
  children,
  disabled,
  pressed,
  title,
}: {
  onClick: () => void;
  children: React.ReactNode;
  disabled?: boolean;
  pressed?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={pressed}
      title={title}
      className={`inline-flex h-8 items-center gap-1.5 rounded-md px-2.5 text-xs font-medium ring-1 transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        pressed ? "bg-slate-900 text-white ring-slate-900" : "bg-white text-slate-700 ring-slate-200 hover:bg-slate-50"
      }`}
    >
      {children}
    </button>
  );
}

export default function Home() {
  const [waypoints, setWaypoints] = useState<Point[]>(WAREHOUSE_ROUTE);
  const [params, setParams] = useState<TrajectoryParams>(DEFAULT_PARAMS);
  const [showTangents, setShowTangents] = useState(false);
  const [compareMode, setCompareMode] = useState(false);

  const request = useMemo<TrajectoryRequest | null>(
    () => (waypoints.length >= 2 ? { ...params, waypoints, samples_per_segment: SAMPLES_PER_SEGMENT } : null),
    [params, waypoints],
  );
  // Modo comparación: Hermite (C¹) frente a cúbico natural (C²) con los mismos waypoints y parametrización
  const compareMethod: Method = params.method === "hermite" ? "natural" : "hermite";
  const compareRequest = useMemo<TrajectoryRequest | null>(
    () => (compareMode && request ? { ...request, method: compareMethod, tension: params.tension } : null),
    [compareMode, request, compareMethod, params.tension],
  );

  const primary = useTrajectory(request);
  const secondary = useTrajectory(compareRequest);
  const anim = useDroneAnimation(primary.data);

  const primaryLabel = SHORT_LABEL[params.method];
  const compareLabel = SHORT_LABEL[compareMethod];
  const compareData = compareMode ? secondary.data : null;
  const error = primary.error ?? (compareMode ? secondary.error : null);

  return (
    <main className="mx-auto max-w-[1400px] px-4 py-6 sm:px-6">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-sky-700">Métodos Numéricos · Interpolación segmentaria</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-900">Trayectorias de drones con splines cúbicos</h1>
          <p className="mt-1 text-sm text-slate-500">
            {METHOD_LABEL[params.method]} sobre {waypoints.length} waypoints: velocidad, curvatura y perfil físicamente factible.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500" aria-live="polite">
          <span
            className={`h-2 w-2 rounded-full ${error ? "bg-rose-500" : primary.loading ? "animate-pulse bg-amber-400" : "bg-emerald-500"}`}
          />
          {error ? "Error" : primary.loading ? "Calculando…" : "Backend conectado"}
        </div>
      </header>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
        <section className="min-w-0 rounded-xl bg-white p-3 ring-1 ring-slate-200">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {anim.playing ? (
              <IconButton onClick={anim.pause} title="Pausa">
                <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden><rect x="2" y="1.5" width="3" height="9" rx="0.5" fill="currentColor" /><rect x="7" y="1.5" width="3" height="9" rx="0.5" fill="currentColor" /></svg>
                Pausa
              </IconButton>
            ) : (
              <IconButton onClick={anim.play} disabled={!primary.data} title="Reproducir">
                <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden><path d="M3 1.5v9l7.5-4.5z" fill="currentColor" /></svg>
                Reproducir
              </IconButton>
            )}
            <IconButton onClick={anim.reset} disabled={!primary.data} title="Volver al inicio">
              <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden><path d="M2.5 6a3.5 3.5 0 1 0 1-2.5M2.5 1.5v2.5H5" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" /></svg>
              Reiniciar
            </IconButton>
            <div className="flex h-8 items-center gap-0.5 rounded-md bg-slate-100 p-0.5" role="radiogroup" aria-label="Velocidad de simulación">
              {SIM_SPEEDS.map((m) => (
                <button
                  key={m}
                  type="button"
                  role="radio"
                  aria-checked={anim.multiplier === m}
                  onClick={() => anim.setMultiplier(m)}
                  className={`h-7 rounded px-2 font-mono text-xs ${anim.multiplier === m ? "bg-white text-slate-900 shadow-sm" : "text-slate-500"}`}
                >
                  {m}×
                </button>
              ))}
            </div>

            <span className="mx-1 hidden h-5 w-px bg-slate-200 sm:block" />

            <IconButton onClick={() => setShowTangents((v) => !v)} pressed={showTangents}>
              Tangentes
            </IconButton>
            <IconButton onClick={() => setCompareMode((v) => !v)} pressed={compareMode} title="Superponer el otro método">
              Comparar con {compareLabel}
            </IconButton>

            <div className="ml-auto flex gap-2">
              <IconButton onClick={() => setWaypoints(WAREHOUSE_ROUTE)}>Ruta de ejemplo</IconButton>
              <IconButton onClick={() => setWaypoints([])} disabled={waypoints.length === 0}>
                Limpiar
              </IconButton>
            </div>
          </div>

          <TrajectoryCanvas
            waypoints={waypoints}
            onChange={setWaypoints}
            result={primary.data}
            compare={compareData}
            showTangents={showTangents}
            vCruise={params.limits.v_cruise}
            droneRef={anim.droneRef}
          />

          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 px-1">
            <CanvasLegend
              vCruise={params.limits.v_cruise}
              compareLabel={compareLabel}
              showCompare={compareData !== null}
              showTangents={showTangents && primary.data !== null}
            />
            <span className="text-[11px] text-slate-400">Clic: agregar · Arrastrar: mover · Clic derecho: borrar</span>
          </div>
          {error && (
            <p role="alert" className="mt-2 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-700 ring-1 ring-rose-200">
              {error}
            </p>
          )}

          <div className="mt-3">
            <StatsPanel result={primary.data} compare={compareData} compareLabel={compareLabel} drone={anim.drone} />
          </div>
        </section>

        <aside className="self-start rounded-xl bg-white p-4 ring-1 ring-slate-200 lg:sticky lg:top-4">
          <ControlPanel params={params} onChange={setParams} />
        </aside>
      </div>

      <section className="mt-5">
        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Perfiles a lo largo de la trayectoria</h2>
        <ProfileCharts
          result={primary.data}
          compare={compareData}
          primaryLabel={primaryLabel}
          compareLabel={compareLabel}
          markerS={anim.drone?.s ?? null}
        />
      </section>
    </main>
  );
}

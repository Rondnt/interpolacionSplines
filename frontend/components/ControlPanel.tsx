"use client";

import { METHOD_LABEL, PARAM_LABEL, type Limits, type Method, type Parametrization, type TrajectoryParams } from "@/lib/types";

interface SegmentedProps<T extends string> {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}

function Segmented<T extends string>({ value, options, onChange, label }: SegmentedProps<T>) {
  return (
    <div role="radiogroup" aria-label={label} className="grid grid-flow-col auto-cols-fr gap-1 rounded-lg bg-slate-100 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
            value === o.value ? "bg-white text-slate-900 shadow-sm ring-1 ring-slate-200" : "text-slate-500 hover:text-slate-800"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  disabled?: boolean;
  onChange: (v: number) => void;
}

function Slider({ label, value, min, max, step, unit, disabled, onChange }: SliderProps) {
  return (
    <label className={`block ${disabled ? "opacity-40" : ""}`}>
      <div className="mb-1 flex items-baseline justify-between text-xs">
        <span className="text-slate-600">{label}</span>
        <span className="font-mono tabular-nums text-slate-900">
          {value.toFixed(step < 0.1 ? 2 : 1)}
          {unit && <span className="ml-0.5 text-slate-400">{unit}</span>}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-sky-600"
      />
    </label>
  );
}

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <div>
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{title}</h3>
        {hint && <p className="mt-0.5 text-[11px] leading-snug text-slate-400">{hint}</p>}
      </div>
      {children}
    </div>
  );
}

const METHOD_HINT: Record<Method, string> = {
  hermite: "Tramos cúbicos con tangentes estimadas. Local y C¹: p″ salta en los nodos.",
  natural: "Sistema tridiagonal (Thomas). C² global, con M₀ = Mₙ = 0 en los extremos.",
  clamped: "Como el natural, pero fija las pendientes en los extremos.",
};

const PARAM_HINT: Record<Parametrization, string> = {
  uniform: "tₖ = k. Puede producir bucles si los waypoints están a distancias muy distintas.",
  chord: "Δtₖ = ‖Pₖ₊₁ − Pₖ‖. El parámetro aproxima la distancia recorrida.",
  centripetal: "Δtₖ = ‖Pₖ₊₁ − Pₖ‖^½. Evita bucles y cúspides (recomendada).",
};

interface Props {
  params: TrajectoryParams;
  onChange: (p: TrajectoryParams) => void;
}

export default function ControlPanel({ params, onChange }: Props) {
  const set = <K extends keyof TrajectoryParams>(key: K, value: TrajectoryParams[K]) => onChange({ ...params, [key]: value });
  const setLimit = (key: keyof Limits, value: number) => set("limits", { ...params.limits, [key]: value });

  return (
    <div className="space-y-5">
      <Section title="Método" hint={METHOD_HINT[params.method]}>
        <Segmented<Method>
          label="Método de interpolación"
          value={params.method}
          onChange={(v) => set("method", v)}
          options={(Object.keys(METHOD_LABEL) as Method[]).map((m) => ({
            value: m,
            label: m === "hermite" ? "Hermite" : m === "natural" ? "Natural" : "Sujeto",
          }))}
        />
      </Section>

      <Section title="Tensión c" hint="Escala las tangentes por (1 − c). c = 0 ⇒ Catmull-Rom; c = 1 ⇒ tramos rectos con parada en cada waypoint.">
        <Slider
          label={params.method === "hermite" ? "Tensión" : "Tensión (solo Hermite)"}
          value={params.tension}
          min={0}
          max={1}
          step={0.05}
          disabled={params.method !== "hermite"}
          onChange={(v) => set("tension", v)}
        />
      </Section>

      <Section title="Parametrización" hint={PARAM_HINT[params.parametrization]}>
        <Segmented<Parametrization>
          label="Parametrización"
          value={params.parametrization}
          onChange={(v) => set("parametrization", v)}
          options={(Object.keys(PARAM_LABEL) as Parametrization[]).map((p) => ({ value: p, label: PARAM_LABEL[p] }))}
        />
      </Section>

      <Section title="Límites físicos" hint="v ≤ min(v_crucero, √(a_lat/κ)) y |dv/dt| ≤ a_long. Parte y llega en reposo.">
        <div className="space-y-3">
          <Slider label="Velocidad de crucero" unit="m/s" value={params.limits.v_cruise} min={0.5} max={15} step={0.5} onChange={(v) => setLimit("v_cruise", v)} />
          <Slider label="Aceleración lateral máx." unit="m/s²" value={params.limits.a_lat_max} min={0.2} max={8} step={0.1} onChange={(v) => setLimit("a_lat_max", v)} />
          <Slider label="Aceleración longitudinal máx." unit="m/s²" value={params.limits.a_long_max} min={0.2} max={8} step={0.1} onChange={(v) => setLimit("a_long_max", v)} />
        </div>
      </Section>
    </div>
  );
}

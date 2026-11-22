// Tipos que reflejan los modelos Pydantic de backend/api/schemas.py

export type Method = "hermite" | "natural" | "clamped";
export type Parametrization = "uniform" | "chord" | "centripetal";
export type Point = [number, number];

export interface Limits {
  v_cruise: number; // m/s
  a_lat_max: number; // m/s²
  a_long_max: number; // m/s²
}

export interface TrajectoryParams {
  method: Method;
  tension: number;
  parametrization: Parametrization;
  limits: Limits;
}

export interface TrajectoryRequest extends TrajectoryParams {
  waypoints: Point[];
  samples_per_segment: number;
  end_tangents?: [Point, Point] | null;
}

export interface Stats {
  length: number; // m
  time_estimate: number; // s
  max_curvature: number; // 1/m
}

export interface TrajectoryResponse {
  points: Point[];
  t: number[];
  s: number[]; // longitud de arco acumulada [m]
  speed: number[]; // ‖p'(t)‖ paramétrica
  acceleration: number[]; // ‖p''(t)‖ paramétrica
  curvature: number[]; // κ [1/m]
  v_max: number[]; // límite por curvatura [m/s]
  v_profile: number[]; // perfil factible [m/s]
  time: number[]; // tiempo acumulado [s]
  tangents: Point[]; // tangentes en los nodos
  knots_index: number[]; // índices de `points` que son waypoints
  stats: Stats;
}

/** Estado instantáneo del dron durante la animación. */
export interface DroneState {
  x: number;
  y: number;
  heading: number; // rad, sentido antihorario desde +x
  s: number; // m recorridos
  v: number; // m/s
  time: number; // s simulados
}

export const METHOD_LABEL: Record<Method, string> = {
  hermite: "Hermite (C¹)",
  natural: "Cúbico natural (C²)",
  clamped: "Cúbico sujeto (C²)",
};

export const PARAM_LABEL: Record<Parametrization, string> = {
  uniform: "Uniforme",
  chord: "Cuerda",
  centripetal: "Centrípeta",
};

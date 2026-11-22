import type { TrajectoryRequest, TrajectoryResponse } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface ValidationItem {
  msg?: string;
  loc?: (string | number)[];
}

/** Convierte el `detail` de FastAPI (string o lista de errores de Pydantic) en un mensaje legible. */
function formatDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = (detail as ValidationItem[]).map((d) => {
      const msg = (d.msg ?? "Solicitud inválida").replace(/^Value error, /, "");
      const field = d.loc?.filter((p) => p !== "body").join(".");
      return field && !msg.startsWith("Los waypoints") ? `${field}: ${msg}` : msg;
    });
    return msgs.join(" · ");
  }
  return null;
}

export async function fetchTrajectory(req: TrajectoryRequest, signal?: AbortSignal): Promise<TrajectoryResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/trajectory`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(`No se pudo conectar con el backend en ${API_URL}. ¿Está corriendo uvicorn?`, 0);
  }
  if (!res.ok) {
    const body: unknown = await res.json().catch(() => null);
    const detail = body && typeof body === "object" && "detail" in body ? (body as { detail: unknown }).detail : null;
    throw new ApiError(formatDetail(detail) ?? `Error ${res.status} del servidor`, res.status);
  }
  return (await res.json()) as TrajectoryResponse;
}

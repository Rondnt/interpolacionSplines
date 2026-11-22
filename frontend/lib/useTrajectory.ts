"use client";

import { useEffect, useRef, useState } from "react";
import { fetchTrajectory } from "./api";
import type { TrajectoryRequest, TrajectoryResponse } from "./types";

interface TrajectoryState {
  data: TrajectoryResponse | null;
  error: string | null;
  loading: boolean;
}

/**
 * Pide la trayectoria al backend cada vez que cambia la solicitud.
 *
 * - Limita la frecuencia a una solicitud cada `interval` ms (throttle con disparo final),
 *   así la curva se actualiza mientras se arrastra un punto y no solo al soltarlo.
 * - No cancela solicitudes en vuelo: si el servidor tarda más que `interval`, cancelar
 *   impediría que llegara alguna respuesta durante el arrastre. En su lugar se numeran
 *   y se descartan las respuestas más viejas que la última aplicada.
 */
export function useTrajectory(req: TrajectoryRequest | null, interval = 150): TrajectoryState {
  const [state, setState] = useState<TrajectoryState>({ data: null, error: null, loading: false });
  const seqRef = useRef(0);
  const appliedRef = useRef(0);
  const lastFireRef = useRef(-Infinity);
  const key = req ? JSON.stringify(req) : null;

  useEffect(() => {
    if (key === null) {
      appliedRef.current = ++seqRef.current; // invalida respuestas pendientes
      setState({ data: null, error: null, loading: false });
      return;
    }
    const wait = Math.max(0, interval - (performance.now() - lastFireRef.current));
    const id = window.setTimeout(() => {
      lastFireRef.current = performance.now();
      const seq = ++seqRef.current;
      setState((s) => ({ ...s, loading: true }));
      fetchTrajectory(JSON.parse(key) as TrajectoryRequest)
        .then((data) => {
          if (seq <= appliedRef.current) return;
          appliedRef.current = seq;
          setState({ data, error: null, loading: seq !== seqRef.current });
        })
        .catch((err: unknown) => {
          if (seq <= appliedRef.current) return;
          appliedRef.current = seq;
          const message = err instanceof Error ? err.message : String(err);
          setState((s) => ({ data: s.data, error: message, loading: seq !== seqRef.current }));
        });
    }, wait);
    return () => window.clearTimeout(id);
  }, [key, interval]);

  return state;
}

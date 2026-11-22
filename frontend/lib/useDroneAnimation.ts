"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { droneAt, interp } from "./numeric";
import type { DroneState, TrajectoryResponse } from "./types";

const MARKER_INTERVAL_MS = 80; // frecuencia de actualización del marcador en las gráficas

/**
 * Anima el dron sobre la trayectoria a velocidad física (perfil v(s) del backend).
 *
 * - `droneRef` se actualiza en cada frame (lo lee el canvas sin re-renderizar React).
 * - `drone` es una copia con menor frecuencia, para el marcador de las gráficas y las estadísticas.
 */
export function useDroneAnimation(result: TrajectoryResponse | null) {
  const [playing, setPlaying] = useState(false);
  const [multiplier, setMultiplier] = useState(1);
  const [drone, setDrone] = useState<DroneState | null>(null);
  const droneRef = useRef<DroneState | null>(null);
  const simTimeRef = useRef(0);

  // Si cambia la trayectoria (se movió un punto), se conserva la distancia recorrida.
  useEffect(() => {
    if (!result) {
      droneRef.current = null;
      simTimeRef.current = 0;
      setDrone(null);
      setPlaying(false);
      return;
    }
    const L = result.s[result.s.length - 1];
    const s = Math.min(droneRef.current?.s ?? 0, L);
    simTimeRef.current = interp(s, result.s, result.time);
    const state = droneAt(result, simTimeRef.current);
    droneRef.current = state;
    setDrone(state);
  }, [result]);

  useEffect(() => {
    if (!playing || !result) return;
    let raf = 0;
    let last = performance.now();
    let lastMarker = 0;
    const T = result.time[result.time.length - 1];

    const step = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000); // evita saltos si la pestaña estuvo inactiva
      last = now;
      simTimeRef.current = Math.min(T, simTimeRef.current + dt * multiplier);
      const state = droneAt(result, simTimeRef.current);
      droneRef.current = state;
      const done = simTimeRef.current >= T;
      if (done || now - lastMarker > MARKER_INTERVAL_MS) {
        lastMarker = now;
        setDrone(state);
      }
      if (done) setPlaying(false);
      else raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, result, multiplier]);

  const play = useCallback(() => {
    if (!result) return;
    const T = result.time[result.time.length - 1];
    if (simTimeRef.current >= T - 1e-9) simTimeRef.current = 0; // al final: reinicia
    setPlaying(true);
  }, [result]);

  const pause = useCallback(() => setPlaying(false), []);

  const reset = useCallback(() => {
    setPlaying(false);
    simTimeRef.current = 0;
    if (result) {
      const state = droneAt(result, 0);
      droneRef.current = state;
      setDrone(state);
    }
  }, [result]);

  return { playing, play, pause, reset, multiplier, setMultiplier, drone, droneRef };
}

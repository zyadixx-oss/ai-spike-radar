import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") || "";

export async function api<T = any>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}/api${path}`, init);
  if (!r.ok) throw new Error(`${r.status} ${await r.text().catch(() => "")}`.slice(0, 300));
  return r.json();
}

export function useApi<T = any>(path: string | null, refreshMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const load = useCallback(async () => {
    if (!path) return;
    setLoading(true);
    try {
      setData(await api<T>(path));
      setError(null);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }, [path]);
  useEffect(() => {
    load();
    if (refreshMs > 0) {
      const id = setInterval(load, refreshMs);
      return () => clearInterval(id);
    }
  }, [load, refreshMs]);
  return { data, error, loading, reload: load };
}

export function wsUrl(symbol?: string) {
  const base = API_BASE || window.location.origin;
  const u = new URL(base.replace(/^http/, "ws") + "/ws");
  if (symbol) u.searchParams.set("symbol", symbol);
  return u.toString();
}

/** Live WebSocket with automatic reconnect. */
export function useLive(onMessage: (m: any) => void, symbol?: string) {
  const cb = useRef(onMessage);
  cb.current = onMessage;
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let ws: WebSocket | null = null;
    let stop = false;
    let retry = 0;
    let timer: any;
    const connect = () => {
      ws = new WebSocket(wsUrl(symbol));
      ws.onopen = () => {
        setConnected(true);
        retry = 0;
      };
      ws.onmessage = (ev) => {
        try {
          cb.current(JSON.parse(ev.data));
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!stop) timer = setTimeout(connect, Math.min(15000, 500 * 2 ** retry++));
      };
    };
    connect();
    return () => {
      stop = true;
      clearTimeout(timer);
      ws?.close();
    };
  }, [symbol]);
  return connected;
}

/** Wrap in Unicode LEFT-TO-RIGHT ISOLATE so numbers, signs and intervals render correctly inside Arabic (RTL) text. */
export const ltr = (s: string) => `\u2066${s}\u2069`;
export const fmtPct = (x: any, d = 2) => (typeof x === "number" && isFinite(x) ? ltr(`${(x * 100).toFixed(d)}%`) : "—");
export const fmtNum = (x: any, d = 3) => (typeof x === "number" && isFinite(x) ? ltr(x.toFixed(d)) : "—");
export const fmtInt = (x: any) => (typeof x === "number" && isFinite(x) ? ltr(x.toLocaleString("en-US")) : "—");
export const fmtCI = (a: any, b: any, d = 4) =>
  typeof a === "number" && typeof b === "number" ? ltr(`[${a.toFixed(d)}, ${b.toFixed(d)}]`) : "—";
export const fmtTime = (epoch: any) =>
  typeof epoch === "number" && epoch > 0 ? ltr(new Date(epoch * 1000).toISOString().replace("T", " ").slice(0, 19)) : "—";
export const fmtAgo = (sec: any) => {
  if (typeof sec !== "number" || !isFinite(sec)) return "—";
  if (sec < 90) return `${Math.round(sec)}s`;
  if (sec < 5400) return `${Math.round(sec / 60)}m`;
  if (sec < 172800) return `${(sec / 3600).toFixed(1)}h`;
  return `${(sec / 86400).toFixed(1)}d`;
};

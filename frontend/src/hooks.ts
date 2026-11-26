import { useApi } from "./api";

export type Sym = { symbol: string; display_name: string; family: string; n: number; direction: number; available: boolean };

export function useSymbols() {
  const { data } = useApi<{ symbols: Sym[]; missing: string[] }>("/symbols", 60000);
  return { symbols: data?.symbols || [], missing: data?.missing || [] };
}

export const HORIZONS = ["10s", "30s", "1m", "3m", "5m", "15m", "50t", "100t", "200t"];
export const TIMEFRAMES = ["tick", "5s", "10s", "15s", "30s", "M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H4", "D1"];
export const TF_SECONDS: Record<string, number> = {
  tick: 0, "5s": 5, "10s": 10, "15s": 15, "30s": 30, M1: 60, M2: 120, M3: 180, M5: 300, M10: 600, M15: 900, M30: 1800,
  H1: 3600, H4: 14400, D1: 86400,
};

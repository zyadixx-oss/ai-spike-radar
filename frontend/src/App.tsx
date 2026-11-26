import { useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { useApi } from "./api";
import { useI18n } from "./i18n";
import Dashboard from "./pages/Dashboard";
import Radar from "./pages/Radar";
import Charts from "./pages/Charts";
import SpikeHistory from "./pages/SpikeHistory";
import Arena from "./pages/Arena";
import Backtesting from "./pages/Backtesting";
import TimeframeRankings from "./pages/TimeframeRankings";
import ModelRankings from "./pages/ModelRankings";
import Predictions from "./pages/Predictions";
import Shadow from "./pages/Shadow";
import Health from "./pages/Health";
import Reports from "./pages/Reports";

const NAV: [string, string, string][] = [
  ["/", "nav_dashboard", "◧"], ["/radar", "nav_radar", "◉"], ["/charts", "nav_charts", "▤"],
  ["/spikes", "nav_spikes", "⚡"], ["/arena", "nav_arena", "⚔"], ["/backtesting", "nav_backtest", "⟲"],
  ["/timeframes", "nav_tf", "⏱"], ["/models", "nav_models", "◆"], ["/predictions", "nav_predictions", "☰"],
  ["/shadow", "nav_shadow", "◌"], ["/health", "nav_health", "♥"], ["/reports", "nav_reports", "▦"],
];

function SafetyBar() {
  const { t } = useI18n();
  const { data } = useApi<any>("/safety", 60000);
  const flags = data ? ["ready_for_live", "live_allowed", "opened_trades", "LIVE_TRADING"] : [];
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-amber-900/50 bg-amber-950/30 px-4 py-1.5 text-[11px] text-amber-300">
      <span className="font-semibold">⛨ {t("research_only")}</span>
      {flags.map((f) => (
        <span key={f} className="font-mono">{f}=<b className={data[f] ? "text-red-400" : "text-emerald-400"}>{String(data[f])}</b></span>
      ))}
    </div>
  );
}

export default function App() {
  const { t, lang, setLang } = useI18n();
  const [open, setOpen] = useState(false);
  return (
    <div className="min-h-screen">
      <SafetyBar />
      <div className="flex">
        <aside
          className={`fixed inset-y-0 z-30 w-64 transform border-e border-line bg-panel2 transition lg:static lg:translate-x-0 ${
            open ? "translate-x-0" : lang === "ar" ? "translate-x-full" : "-translate-x-full"
          } start-0`}
        >
          <div className="flex items-center gap-2 border-b border-line px-4 py-4">
            <svg viewBox="0 0 32 32" className="h-8 w-8 shrink-0"><circle cx="16" cy="16" r="14" fill="#0b0f17" stroke="#38bdf8" strokeWidth="2" /><path d="M6 20 L12 18 L15 6 L18 22 L21 16 L26 15" stroke="#f97316" strokeWidth="2.4" fill="none" /></svg>
            <div>
              <div className="text-sm font-bold">{t("app_title")}</div>
              <div className="text-[11px] text-muted">{t("app_sub")}</div>
            </div>
          </div>
          <nav className="space-y-0.5 p-2">
            {NAV.map(([to, key, icon]) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${isActive ? "bg-accent/15 text-accent" : "text-ink/80 hover:bg-white/5"}`
                }
              >
                <span className="w-4 text-center opacity-80">{icon}</span>
                {t(key)}
              </NavLink>
            ))}
          </nav>
          <div className="mt-2 px-4 text-[11px] leading-relaxed text-muted">{t("disclaimer")}</div>
        </aside>
        {open && <div className="fixed inset-0 z-20 bg-black/50 lg:hidden" onClick={() => setOpen(false)} />}
        <main className="min-w-0 flex-1">
          <header className="sticky top-0 z-10 flex items-center justify-between gap-2 border-b border-line bg-bg/90 px-4 py-2 backdrop-blur">
            <button className="rounded-md border border-line px-2 py-1 text-sm lg:hidden" onClick={() => setOpen(true)} aria-label="menu">☰</button>
            <div className="truncate text-sm text-muted">{t("app_sub")}</div>
            <div className="flex gap-1">
              {(["ar", "en"] as const).map((l) => (
                <button key={l} onClick={() => setLang(l)} className={`rounded-md px-2 py-1 text-xs ${lang === l ? "bg-accent/20 text-accent" : "text-muted hover:text-ink"}`}>
                  {l === "ar" ? "العربية" : "English"}
                </button>
              ))}
            </div>
          </header>
          <div className="mx-auto max-w-[1500px] p-4">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/radar" element={<Radar />} />
              <Route path="/charts" element={<Charts />} />
              <Route path="/spikes" element={<SpikeHistory />} />
              <Route path="/arena" element={<Arena />} />
              <Route path="/backtesting" element={<Backtesting />} />
              <Route path="/timeframes" element={<TimeframeRankings />} />
              <Route path="/models" element={<ModelRankings />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/shadow" element={<Shadow />} />
              <Route path="/health" element={<Health />} />
              <Route path="/reports" element={<Reports />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}

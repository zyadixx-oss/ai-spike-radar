import type { ReactNode } from "react";
import { useI18n } from "../i18n";

export const LEVELS = ["gray", "green", "yellow", "orange", "red"] as const;
export const levelColor: Record<string, string> = {
  gray: "#6b7280", green: "#22c55e", yellow: "#eab308", orange: "#f97316", red: "#ef4444",
};

export function Card({ title, children, right, className = "" }: { title?: ReactNode; children: ReactNode; right?: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-line bg-panel/80 p-4 shadow-sm ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold tracking-wide text-ink">{title}</h2>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, tone }: { label: ReactNode; value: ReactNode; sub?: ReactNode; tone?: string }) {
  return (
    <div className="min-w-0 rounded-xl border border-line bg-panel2 p-3">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 truncate font-mono text-xl font-semibold" style={tone ? { color: tone } : undefined}
        title={typeof value === "string" ? value : undefined}>{value}</div>
      {sub && <div className="mt-0.5 truncate text-xs text-muted" title={typeof sub === "string" ? sub : undefined}>{sub}</div>}
    </div>
  );
}

export function LevelBadge({ level, small }: { level?: string; small?: boolean }) {
  const { t } = useI18n();
  const l = level || "gray";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 ${small ? "py-0 text-[11px]" : "py-0.5 text-xs"} font-medium`}
      style={{ borderColor: levelColor[l] + "88", color: levelColor[l], background: levelColor[l] + "1a" }}
      title={t(`lvl_${l}`)}
    >
      <span className="h-2 w-2 rounded-full" style={{ background: levelColor[l] }} />
      {small ? l.toUpperCase() : t(`lvl_${l}`)}
    </span>
  );
}

export function Pill({ children, tone = "#38bdf8" }: { children: ReactNode; tone?: string }) {
  return (
    <span className="inline-block rounded-md px-1.5 py-0.5 text-[11px] font-medium" style={{ color: tone, background: tone + "1f" }}>
      {children}
    </span>
  );
}

export function Empty({ children }: { children?: ReactNode }) {
  const { t } = useI18n();
  return <div className="rounded-lg border border-dashed border-line p-6 text-center text-sm text-muted">{children || t("no_data")}</div>;
}

export function Loader({ error, loading }: { error?: string | null; loading?: boolean }) {
  const { t } = useI18n();
  if (error) return <div className="rounded-lg border border-red-900 bg-red-950/40 p-3 text-sm text-red-300">{t("error")}: {error}</div>;
  if (loading) return <div className="p-4 text-sm text-muted">{t("loading")}</div>;
  return null;
}

export function Table({ head, rows, dense }: { head: ReactNode[]; rows: ReactNode[][]; dense?: boolean }) {
  return (
    <div className="-mx-1 overflow-x-auto">
      <table className="w-full min-w-max border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-start text-xs text-muted">
            {head.map((h, i) => (
              <th key={i} className="whitespace-nowrap px-2 py-2 text-start font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line/60 hover:bg-white/[0.02]">
              {r.map((c, j) => (
                <td key={j} className={`whitespace-nowrap px-2 ${dense ? "py-1" : "py-2"} align-top`}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ProbBar({ p, base, level }: { p?: number | null; base?: number | null; level?: string }) {
  const max = Math.max(p || 0, base || 0, 0.0001) * 1.3;
  const w = Math.min(100, ((p || 0) / max) * 100);
  const b = Math.min(100, ((base || 0) / max) * 100);
  return (
    <div className="relative h-2 w-28 overflow-hidden rounded bg-line" title={`p=${p} base=${base}`}>
      <div className="h-full rounded" style={{ width: `${w}%`, background: levelColor[level || "gray"] }} />
      {base ? <div className="absolute top-0 h-full w-0.5 bg-white/70" style={{ insetInlineStart: `${b}%` }} /> : null}
    </div>
  );
}

export function Select({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; label?: string }) {
  return (
    <label className="inline-flex items-center gap-2 text-xs text-muted">
      {label}
      <select
        className="rounded-md border border-line bg-panel2 px-2 py-1.5 text-sm text-ink focus:outline-none focus:ring-1 focus:ring-accent"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

export function Btn({ children, onClick, active, disabled }: { children: ReactNode; onClick?: () => void; active?: boolean; disabled?: boolean }) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className={`rounded-md border px-2.5 py-1.5 text-xs font-medium transition disabled:opacity-40 ${
        active ? "border-accent bg-accent/15 text-accent" : "border-line bg-panel2 text-ink hover:border-accent/60"
      }`}
    >
      {children}
    </button>
  );
}

export function Verdict({ ok, children }: { ok: boolean | null | undefined; children: ReactNode }) {
  const tone = ok ? "#22c55e" : ok === false ? "#f59e0b" : "#8b95a7";
  return (
    <div className="rounded-lg border p-3 text-sm" style={{ borderColor: tone + "66", background: tone + "12", color: tone }}>
      {children}
    </div>
  );
}

export function Json({ data }: { data: any }) {
  return <pre className="max-h-80 overflow-auto rounded-lg bg-black/40 p-3 text-[11px] leading-relaxed text-muted">{JSON.stringify(data, null, 2)}</pre>;
}

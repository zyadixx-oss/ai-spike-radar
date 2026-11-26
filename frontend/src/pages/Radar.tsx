import { useMemo, useState } from "react";
import { api, fmtNum, fmtPct, fmtTime, useApi, useLive } from "../api";
import { Btn, Card, Empty, LEVELS, LevelBadge, Loader, Pill, ProbBar, Select, Table } from "../components/ui";
import { HORIZONS } from "../hooks";
import { useI18n, useTx } from "../i18n";

const ORDER: Record<string, number> = { gray: 0, green: 1, yellow: 2, orange: 3, red: 4 };

export default function Radar() {
  const { t } = useI18n();
  const tx = useTx();
  const initial = useApi<any>("/radar", 0);
  const [rows, setRows] = useState<any[] | null>(null);
  const [horizon, setHorizon] = useState("best");
  const [sel, setSel] = useState<string | null>(null);
  const connected = useLive((m) => {
    if (m.type === "radar") setRows(m.data);
    if (m.type === "hello" && m.radar) setRows(m.radar);
  });
  const data = rows ?? initial.data?.rows ?? null;

  const sorted = useMemo(() => {
    if (!data) return [];
    return [...data]
      .map((r) => {
        const key = horizon === "best" ? r.top_horizon : horizon;
        const h = r.horizons?.[key] || {};
        return { ...r, key, h, lift: h.p && h.base ? h.p / h.base : null };
      })
      .sort((a, b) => (ORDER[b.h.level || "gray"] - ORDER[a.h.level || "gray"]) || ((b.lift || 0) - (a.lift || 0)));
  }, [data, horizon]);

  return (
    <div className="space-y-4">
      <Card
        title={<span className="flex items-center gap-2">{t("nav_radar")} <Pill tone={connected ? "#22c55e" : "#ef4444"}>{connected ? t("live") : t("disconnected")}</Pill></span>}
        right={
          <Select label={t("horizon")} value={horizon} onChange={setHorizon}
            options={[{ value: "best", label: tx("الأقوى", "Strongest") }, ...HORIZONS.map((h) => ({ value: h, label: h }))]} />
        }
      >
        <div className="mb-3 flex flex-wrap gap-2 text-xs text-muted">
          {LEVELS.map((l) => <LevelBadge key={l} level={l} />)}
        </div>
        <p className="mb-3 text-xs text-muted">
          {tx("حدود الألوان مستخرجة من بيانات التحقق (الحد الأدنى لفاصل ثقة ويلسون للدقة مقارنة بالمعدل الطبيعي)، والأحمر لا يظهر إلا لنموذج اجتاز اختبار البيانات غير المرئية.",
              "Colour cut-offs are derived from validation data (Wilson lower bound of precision vs base rate); red only appears for a model that passed the unseen-holdout gate.")}
        </p>
        <Loader error={initial.error} loading={!data && initial.loading} />
        {data && !data.length && <Empty>{tx("لا توجد بيانات حية بعد. تحقق من صفحة حالة النظام.", "No live data yet — check System Health.")}</Empty>}
        {sorted.length > 0 && (
          <Table
            head={[t("symbol"), t("level"), t("horizon"), t("probability"), t("base_rate"), t("lift"), t("model"), t("direction"), tx("آخر سعر", "Last"), ""]}
            rows={sorted.map((r) => [
              <b>{r.display_name}</b>,
              <LevelBadge level={r.h.level} />,
              r.key,
              <div className="flex items-center gap-2"><ProbBar p={r.h.p} base={r.h.base} level={r.h.level} /><span className="font-mono text-xs">{fmtPct(r.h.p)}</span></div>,
              <span className="font-mono text-xs">{fmtPct(r.h.base)}</span>,
              <span className="font-mono text-xs">{r.lift ? `×${r.lift.toFixed(2)}` : "—"}</span>,
              <span className="text-xs">{r.h.model}{r.h.approved ? <Pill tone="#22c55e">{t("approved")}</Pill> : null}</span>,
              (r.direction ?? (r.display_name?.startsWith("Boom") ? 1 : -1)) > 0 ? `▲ ${t("up")}` : `▼ ${t("down")}`,
              <span className="font-mono text-xs">{fmtNum(r.last_quote, 3)}</span>,
              <Btn onClick={() => setSel(sel === r.symbol ? null : r.symbol)} active={sel === r.symbol}>{tx("تفاصيل", "Details")}</Btn>,
            ])}
          />
        )}
      </Card>
      {sel && <SymbolDetail symbol={sel} />}
    </div>
  );
}

function SymbolDetail({ symbol }: { symbol: string }) {
  const { t } = useI18n();
  const tx = useTx();
  const { data, error } = useApi<any>(`/radar/${symbol}`, 3000);
  if (!data) return <Loader error={error} loading />;
  return (
    <Card title={`${data.display_name} — ${tx("كل الآفاق", "all horizons")}`} right={<span className="font-mono text-xs text-muted">{fmtTime(data.last_epoch)}</span>}>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {Object.values<any>(data.horizons || {}).map((h) => (
          <div key={h.horizon} className="rounded-lg border border-line bg-panel2 p-3 text-sm">
            <div className="mb-2 flex items-center justify-between">
              <b>{h.horizon}</b>
              <LevelBadge level={h.level} />
            </div>
            <div className="grid grid-cols-2 gap-1 text-xs">
              <span className="text-muted">{t("probability")}</span><span className="font-mono">{fmtPct(h.probability)}</span>
              <span className="text-muted">{t("base_rate")}</span><span className="font-mono">{fmtPct(h.base_rate)}</span>
              <span className="text-muted">{t("direction")}</span><span>{h.direction > 0 ? `▲ ${t("up")}` : `▼ ${t("down")}`}</span>
              <span className="text-muted">{t("expires")}</span><span className="font-mono">{h.horizon_kind === "tick" ? `${h.horizon_size} ticks` : `${h.horizon_size}s`}</span>
              <span className="text-muted">{t("model")}</span><span>{h.model_name} {h.approved ? "✓" : ""}</span>
              <span className="text-muted">{t("best_tf")}</span><span>{h.best_timeframe?.timeframe || tx("لا يوجد فريم مُتحقق", "none validated")}</span>
              <span className="text-muted">{t("expected_size")}</span>
              <span className="font-mono">{h.expected_size?.median_abs_move != null ? `${fmtNum(h.expected_size.median_abs_move, 2)} (IQR ${fmtNum(h.expected_size.p25_abs_move, 2)}–${fmtNum(h.expected_size.p75_abs_move, 2)})` : "—"}</span>
              <span className="text-muted">{t("false_alarms")}</span><span className="font-mono">{fmtPct(h.performance?.false_alarm_share_val)}</span>
              <span className="text-muted">{tx("BSS على الاختبار", "Test BSS")}</span><span className="font-mono">{fmtNum(h.performance?.test_bss_vs_base_rate, 4)}</span>
              <span className="text-muted">PR-AUC</span><span className="font-mono">{fmtNum(h.performance?.test_pr_auc, 3)}</span>
            </div>
            {h.error && <div className="mt-2 text-xs text-red-400">{h.error}</div>}
            <div className="mt-2 text-xs">
              <div className="text-muted">{t("reasons")}:</div>
              <ul className="ms-4 list-disc">
                {(h.reasons || []).map((r: any, i: number) => (
                  <li key={i}>{r.text || <><span className="font-mono">{r.feature}</span> = {fmtNum(r.value, 4)} (z {r.z >= 0 ? "+" : ""}{fmtNum(r.z, 1)})</>}</li>
                ))}
              </ul>
            </div>
            {h.shadow?.length > 0 && (
              <div className="mt-2 text-[11px] text-muted">
                {tx("ظلّ:", "Shadow:")} {h.shadow.map((s: any) => `${s.model_name} ${fmtPct(s.probability)}`).join(" · ")}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-3 flex gap-2">
        <Btn onClick={() => api(`/jobs/train?symbol=${symbol}`, { method: "POST" })}>{tx("إعادة التدريب", "Retrain")}</Btn>
        <Btn onClick={() => api(`/jobs/arena?symbol=${symbol}`, { method: "POST" })}>{tx("تشغيل المختبر", "Run arena")}</Btn>
      </div>
    </Card>
  );
}

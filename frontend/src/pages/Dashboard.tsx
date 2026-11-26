import { Link } from "react-router-dom";
import { fmtInt, fmtPct, fmtTime, useApi } from "../api";
import { Card, Empty, LevelBadge, Loader, ProbBar, Stat, Table, Verdict } from "../components/ui";
import { useI18n, useTx } from "../i18n";

export default function Dashboard() {
  const { t } = useI18n();
  const tx = useTx();
  const { data, error, loading } = useApi<any>("/stats/overview", 5000);
  const rep = useApi<any>("/reports/summary", 60000).data;
  if (!data) return <Loader error={error} loading={loading} />;
  const approvedCount = data.approved_models?.length || 0;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Deriv" value={data.deriv_connected ? t("connected") : t("disconnected")} tone={data.deriv_connected ? "#22c55e" : "#ef4444"} />
        <Stat label={tx("المؤشرات المكتشفة", "Discovered symbols")} value={data.symbols} sub="/ 10" />
        <Stat label={t("ticks")} value={fmtInt(data.ticks)} />
        <Stat label={t("spikes")} value={fmtInt(data.spikes)} />
        <Stat label={t("predictions")} value={fmtInt(data.predictions)} sub={`${t("outcomes")}: ${fmtInt(Object.values(data.outcomes || {}).reduce((a: number, b: any) => a + b, 0))}`} />
        <Stat label={tx("نماذج معتمدة", "Approved models")} value={approvedCount} tone={approvedCount ? "#22c55e" : "#8b95a7"} />
      </div>

      <Verdict ok={approvedCount > 0 ? true : rep?.symbols?.some((s: any) => s.q1_better_than_chance?.best_test) ? false : null}>
        {approvedCount > 0
          ? tx(`يوجد ${approvedCount} نموذج اجتاز اختبار البيانات غير المرئية وتفوّق على النماذج المرجعية.`,
               `${approvedCount} model(s) passed the unseen-holdout gate and beat the reference baselines.`)
          : tx("لا يوجد حتى الآن أي نموذج اجتاز اختبار القبول — لا دليل إحصائي على إمكانية التوقع أفضل من المعدل الطبيعي. البحث مستمر.",
               "No model has passed the acceptance gate yet — no statistical evidence of predictability beyond the base rate. Research continues.")}
      </Verdict>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2" title={t("nav_radar")} right={<Link className="text-xs text-accent" to="/radar">→</Link>}>
          {data.radar?.length ? (
            <Table
              head={[t("symbol"), t("level"), tx("أعلى أفق", "Top horizon"), t("probability"), t("base_rate"), t("model")]}
              rows={data.radar.slice(0, 10).map((r: any) => {
                const h = r.horizons?.[r.top_horizon] || {};
                return [
                  <b>{r.display_name}</b>, <LevelBadge level={r.level} />, r.top_horizon,
                  <div className="flex items-center gap-2"><ProbBar p={h.p} base={h.base} level={h.level} /><span className="font-mono text-xs">{fmtPct(h.p)}</span></div>,
                  <span className="font-mono text-xs">{fmtPct(h.base)}</span>, <span className="block max-w-[150px] truncate text-xs text-muted" title={h.model}>{h.model}</span>,
                ];
              })}
            />
          ) : (
            <Empty>{tx("الرادار ينتظر بيانات حية من Deriv", "Radar is waiting for live Deriv data")}</Empty>
          )}
        </Card>
        <Card title={tx("آخر الانفجارات", "Latest spikes")} right={<Link className="text-xs text-accent" to="/spikes">→</Link>}>
          {data.recent_spikes?.length ? (
            <Table dense head={[t("symbol"), t("time"), t("size"), "σ"]}
              rows={data.recent_spikes.map((s: any) => [s.symbol, <span className="font-mono text-xs">{fmtTime(s.epoch)}</span>, s.size_class, (s.magnitude_sigma || 0).toFixed(0)])} />
          ) : <Empty />}
        </Card>
      </div>

      <Card title={tx("أسئلة البحث — الإجابات الحالية", "Research questions — current answers")} right={<Link className="text-xs text-accent" to="/reports">→</Link>}>
        {rep?.symbols?.length ? (
          <Table
            head={[t("symbol"), tx("أفضل من الصدفة؟", "Better than chance?"), t("memoryless"), tx("فريم مُتحقق", "Validated TF"), tx("أنماط قوية", "Robust patterns")]}
            rows={rep.symbols.map((s: any) => [
              <b>{s.display_name}</b>,
              <span className={s.q1_better_than_chance.approved_horizons.length ? "text-emerald-400" : "text-muted"}>{s.q1_better_than_chance.answer}</span>,
              s.q_memoryless == null ? "—" : s.q_memoryless ? t("yes") : t("no"),
              Array.isArray(s.q2_best_timeframe) ? s.q2_best_timeframe.map((x: any) => `${x.horizon}:${x.timeframe}`).join(", ") : <span className="text-muted">—</span>,
              Array.isArray(s.q3_patterns) ? s.q3_patterns.length : 0,
            ])}
          />
        ) : <Empty />}
      </Card>
    </div>
  );
}

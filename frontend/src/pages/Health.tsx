import { api, fmtAgo, fmtInt, fmtNum, fmtTime, useApi } from "../api";
import { Btn, Card, Empty, Loader, Pill, Stat, Table } from "../components/ui";
import { useI18n, useTx } from "../i18n";

const lvlTone: Record<string, string> = { error: "#ef4444", warn: "#f59e0b", info: "#8b95a7" };

export default function Health() {
  const { t } = useI18n();
  const tx = useTx();
  const { data, error, loading } = useApi<any>("/health", 5000);
  const jobs = useApi<any[]>("/jobs?limit=40", 10000);
  if (!data) return <Loader error={error} loading={loading} />;
  const d = data.deriv || {};
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
        <Stat label="Deriv WebSocket" value={d.connected ? t("connected") : t("disconnected")} tone={d.connected ? "#22c55e" : "#ef4444"} sub={d.url} />
        <Stat label={tx("اتصالات / انقطاعات", "Connects / drops")} value={`${d.connects ?? 0} / ${d.disconnects ?? 0}`} sub={d.last_message_at ? `${tx("آخر رسالة", "last msg")} ${fmtAgo(data.now - d.last_message_at)}` : undefined} />
        <Stat label={tx("مدة التشغيل", "Uptime")} value={fmtAgo(data.uptime_sec)} />
        <Stat label={t("predictions")} value={fmtInt(data.predictions)} sub={`${t("outcomes")}: ${fmtInt(data.outcomes)}`} />
        <Stat label={tx("النماذج", "Models")} value={Object.entries(data.models || {}).map(([k, v]) => `${k[0]}:${v}`).join(" ") || "0"} sub={`${t("approved")}: ${data.approved_champions}`} />
        <Stat label={tx("قاعدة البيانات", "Database")} value={data.database} sub={`PyTorch: ${data.torch_available ? "yes" : "no"}`} />
      </div>
      {d.last_error && <div className="rounded-lg border border-red-900 bg-red-950/30 p-3 text-xs text-red-300">Deriv: {d.last_error}</div>}
      <Card title={t("safety_flags")}>
        <div className="flex flex-wrap gap-2 font-mono text-xs">
          {Object.entries<any>(data.safety || {}).filter(([k]) => k !== "allowed_deriv_calls").map(([k, v]) => <Pill key={k} tone={v === true ? "#ef4444" : "#22c55e"}>{k}={String(v)}</Pill>)}
        </div>
        <div className="mt-2 text-xs text-muted">{tx("رسائل Deriv المسموح بها فقط", "Only allowed Deriv calls")}: {(data.safety?.allowed_deriv_calls || []).join(", ")}</div>
      </Card>
      <Card title={tx("البيانات لكل مؤشر", "Data per symbol")}>
        {data.missing_symbols?.length > 0 && <div className="mb-2 text-xs text-amber-300">{tx("غير متاح حاليًا من Deriv", "Not currently offered by Deriv")}: {data.missing_symbols.join(", ")}</div>}
        {data.symbols?.length ? (
          <Table dense head={[t("symbol"), "Code", t("ticks"), tx("التغطية", "Coverage"), tx("أول", "First"), tx("آخر تيك منذ", "Last tick"), tx("فجوات", "Gaps"), tx("ثواني الفجوات", "Gap secs"), t("spikes"), tx("Backfill مكتمل", "Backfill done"), tx("ذاكرة حية", "Live buffer")]}
            rows={data.symbols.map((s: any) => [s.display_name, <span className="font-mono text-xs">{s.symbol}</span>, fmtInt(s.ticks), `${fmtNum(s.coverage_days, 2)} d`,
              <span className="font-mono text-xs">{fmtTime(s.first_epoch)}</span>,
              <span style={{ color: s.last_tick_age_sec != null && s.last_tick_age_sec < 30 ? "#22c55e" : "#f59e0b" }}>{fmtAgo(s.last_tick_age_sec)}</span>,
              s.gaps, fmtInt(s.gap_seconds), fmtInt(s.spikes), s.backfill?.ticks_complete ? "✓" : "…", fmtInt(s.live_buffer)])} />
        ) : <Empty>{tx("لم تُكتشف رموز بعد — بانتظار الاتصال بـ Deriv", "No symbols discovered yet — waiting for Deriv connection")}</Empty>}
      </Card>
      <Card title={tx("المهام المؤتمتة", "Automated jobs")} right={<div className="flex gap-1">{Object.entries<any>(data.jobs || {}).map(([k, v]) => <Pill key={k} tone="#8b95a7">{k}: {v}</Pill>)}<Btn onClick={() => api("/jobs/evaluate", { method: "POST" })}>evaluate</Btn></div>}>
        {jobs.data?.length ? (
          <Table dense head={["ID", tx("النوع", "Type"), "Payload", tx("الحالة", "Status"), tx("محاولات", "Attempts"), tx("بدأ", "Started"), tx("انتهى", "Finished"), tx("خطأ", "Error")]}
            rows={jobs.data.map((j: any) => [j.id, j.type, <span className="font-mono text-[11px]">{j.payload}</span>, j.status, `${j.attempts}/${j.max_attempts}`,
              <span className="font-mono text-[11px]">{fmtTime(j.started_at)}</span>, <span className="font-mono text-[11px]">{fmtTime(j.finished_at)}</span>,
              j.status === "done" && j.last_error
                ? <span className="text-[11px] text-muted" title={j.last_error}>{tx("نجح بعد إعادة المحاولة", "succeeded after retry")}</span>
                : <span className="text-[11px] text-red-300" title={j.last_error}>{String(j.last_error || "").slice(0, 80)}</span>])} />
        ) : <Empty />}
      </Card>
      <Card title={tx("سجل الأحداث", "Event log")}>
        {data.events?.length ? (
          <Table dense head={[t("time"), tx("المستوى", "Level"), tx("المكوّن", "Component"), tx("الرسالة", "Message")]}
            rows={data.events.map((e: any) => [<span className="font-mono text-[11px]">{fmtTime(e.ts)}</span>, <span style={{ color: lvlTone[e.level] }}>{e.level}</span>, e.component,
              <span className="whitespace-normal text-xs">{e.message}</span>])} />
        ) : <Empty />}
      </Card>
    </div>
  );
}

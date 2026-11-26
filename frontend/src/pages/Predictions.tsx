import { useState } from "react";
import { fmtInt, fmtPct, fmtTime, useApi } from "../api";
import { Btn, Card, Empty, LevelBadge, Loader, Pill, Select, Table } from "../components/ui";
import { HORIZONS, useSymbols } from "../hooks";
import { useI18n, useTx } from "../i18n";

export default function Predictions() {
  const { t } = useI18n();
  const tx = useTx();
  const { symbols } = useSymbols();
  const [f, setF] = useState({ symbol: "", horizon: "", role: "champion", level: "", status: "" });
  const [page, setPage] = useState(0);
  const q = new URLSearchParams(Object.entries(f).filter(([, v]) => v) as any);
  q.set("limit", "100");
  q.set("offset", String(page * 100));
  const { data, error, loading } = useApi<any>(`/predictions?${q}`, 10000);
  const set = (k: string) => (v: string) => { setF({ ...f, [k]: v }); setPage(0); };
  const statusTone: Record<string, string> = { hit: "#22c55e", miss: "#ef4444", invalid: "#8b95a7" };
  return (
    <Card title={t("nav_predictions")} right={<Pill tone="#22c55e">{tx("سجل غير قابل للتعديل أو الحذف", "Append-only · cannot be edited or deleted")}</Pill>}>
      <p className="mb-3 text-xs text-muted">
        {tx("كل توقع يُسجَّل قبل معرفة نتيجته، وتُحفظ النتيجة لاحقًا في جدول منفصل. قاعدة البيانات نفسها ترفض أي تعديل أو حذف (Triggers).",
            "Every prediction is stored before its outcome is known; the outcome is written later to a separate table. The database itself rejects any UPDATE or DELETE (triggers).")}
      </p>
      <div className="mb-3 flex flex-wrap gap-2">
        <Select label={t("symbol")} value={f.symbol} onChange={set("symbol")} options={[{ value: "", label: t("all") }, ...symbols.map((s) => ({ value: s.symbol, label: s.display_name }))]} />
        <Select label={t("horizon")} value={f.horizon} onChange={set("horizon")} options={[{ value: "", label: t("all") }, ...HORIZONS.map((h) => ({ value: h, label: h }))]} />
        <Select label={tx("الدور", "Role")} value={f.role} onChange={set("role")} options={["", "champion", "shadow", "baseline"].map((r) => ({ value: r, label: r || t("all") }))} />
        <Select label={t("level")} value={f.level} onChange={set("level")} options={["", "gray", "green", "yellow", "orange", "red"].map((r) => ({ value: r, label: r || t("all") }))} />
        <Select label={tx("النتيجة", "Outcome")} value={f.status} onChange={set("status")} options={["", "hit", "miss", "invalid", "pending"].map((r) => ({ value: r, label: r ? t(r) : t("all") }))} />
      </div>
      <Loader error={error} loading={loading && !data} />
      {data && !data.rows.length && <Empty />}
      {data?.rows?.length > 0 && (
        <>
          <Table dense
            head={["ID", tx("وقت الإصدار", "Issued"), t("symbol"), t("horizon"), t("level"), t("probability"), t("base_rate"), t("model"), tx("الدور", "Role"), tx("النتيجة", "Outcome"), tx("مدة الإنذار", "Lead"), t("reasons")]}
            rows={data.rows.map((p: any) => [
              p.id, <span className="font-mono text-xs">{fmtTime(p.issued_epoch)}</span>, p.symbol, p.horizon, <LevelBadge small level={p.level} />,
              <span className="font-mono">{fmtPct(p.probability)}</span>, <span className="font-mono text-xs">{fmtPct(p.base_rate)}</span>,
              <span className="text-xs">{p.model_name}{p.approved ? " ✓" : ""}</span>, p.role,
              p.outcome_status ? <span style={{ color: statusTone[p.outcome_status] }}>{t(p.outcome_status)}</span> : <span className="text-muted">{t("pending")}</span>,
              p.lead_time_sec != null ? `${Math.round(p.lead_time_sec)}s` : "—",
              <span className="text-[11px] text-muted">{(p.reasons || []).slice(0, 2).map((r: any) => r.text || r.feature).join(", ")}</span>,
            ])} />
          <div className="mt-3 flex items-center gap-2 text-xs">
            <Btn disabled={page === 0} onClick={() => setPage(page - 1)}>‹</Btn>
            <span className="text-muted">{page + 1} / {Math.max(1, Math.ceil(data.total / 100))} · {fmtInt(data.total)}</span>
            <Btn disabled={(page + 1) * 100 >= data.total} onClick={() => setPage(page + 1)}>›</Btn>
          </div>
        </>
      )}
    </Card>
  );
}

import { fmtCI, fmtNum, fmtPct, fmtTime, useApi } from "../api";
import { Card, Empty, Json, Loader, Pill, Table, Verdict } from "../components/ui";
import { useI18n, useTx } from "../i18n";

function Ans({ v, render }: { v: any; render: (x: any[]) => any }) {
  if (Array.isArray(v)) return v.length ? render(v) : <span className="text-muted">—</span>;
  if (v == null) return <span className="text-muted">—</span>;
  return <span className="text-muted">{String(v)}</span>;
}

export default function Reports() {
  const { t } = useI18n();
  const tx = useTx();
  const { data, error, loading } = useApi<any>("/reports/summary", 60000);
  const download = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `spike-radar-report-${Date.now()}.json`;
    a.click();
  };
  return (
    <div className="space-y-4">
      <Card title={t("nav_reports")} right={data && <button className="text-xs text-accent" onClick={download}>⬇ JSON</button>}>
        <p className="text-xs text-muted">{data?.disclaimer} · {fmtTime(data?.generated_at)}</p>
        <Loader error={error} loading={loading && !data} />
        {data && !data.symbols?.length && <Empty />}
      </Card>
      {data?.symbols?.map((s: any) => (
        <Card key={s.symbol} title={s.display_name} right={<span className="text-xs text-muted">{fmtNum(s.data?.days, 1)} d · {s.spikes ?? 0} {t("spikes")}</span>}>
          <div className="space-y-3 text-sm">
            <Verdict ok={s.q1_better_than_chance?.approved_horizons?.length > 0}>
              <b>{tx("1) هل يمكن التوقع أفضل من الصدفة؟", "1) Predictable better than chance?")}</b> {s.q1_better_than_chance?.answer}
              {s.q1_better_than_chance?.best_test && (
                <span className="ms-2 font-mono text-xs">best test BSS {fmtNum(s.q1_better_than_chance.best_test.bss, 4)} {fmtCI(s.q1_better_than_chance.best_test.ci95?.[0], s.q1_better_than_chance.best_test.ci95?.[1])} ({s.q1_better_than_chance.best_test.horizon}, {s.q1_better_than_chance.best_test.model}{s.q1_better_than_chance.best_test.approved ? "" : `, ${t("not_approved")}`})</span>
              )}
            </Verdict>
            <div><b>{tx("هل الانفجارات عديمة الذاكرة؟", "Are spikes memoryless?")}</b>{" "}
              {s.q_memoryless == null ? "—" : s.q_memoryless ? <Pill tone="#22c55e">{t("yes")}</Pill> : <Pill tone="#f59e0b">{t("no")}</Pill>}
              <span className="ms-2 font-mono text-xs text-muted">Weibull k={fmtNum(s.interarrival?.weibull_shape, 3)} · KS p={fmtNum(s.interarrival?.ks_vs_exponential?.p_value, 3)} · hazard χ² p={fmtNum(s.interarrival?.hazard_flatness_chi2?.p_value, 3)}</span>
            </div>
            <div><b>{tx("2) أفضل فريم", "2) Best timeframe")}:</b>{" "}
              <Ans v={s.q2_best_timeframe} render={(x) => x.map((r: any, i: number) => <Pill key={i} tone={r.test_confirmed ? "#22c55e" : "#eab308"}>{r.horizon}: {r.timeframe} (val ×{fmtNum(r.best_val_lift, 2)}, test ×{fmtNum(r.test_lift, 2)})</Pill>)} />
            </div>
            <div><b>{tx("3) الأنماط التي تسبق الانفجارات", "3) Patterns that precede spikes")}:</b>{" "}
              <Ans v={s.q3_patterns} render={(x) => <Table dense head={["Window", "Stat", "AUC", "q"]} rows={x.map((p: any) => [p.window, p.stat, fmtNum(p.auc, 3), fmtNum(p.q_value, 4)])} />} />
            </div>
            <div><b>{tx("4) أفضل استراتيجية", "4) Best strategy")}:</b>{" "}
              <Ans v={s.q4_best_strategy} render={(x) => <Table dense head={[t("horizon"), "Family", "TF", "Rule", "Val lift", "Test lift", ""]}
                rows={x.map((r: any) => [r.horizon, r.family, r.timeframe, `${r.feature} ${r.side} ${r.q ?? ""}`, fmtNum(r.val_lift, 2), fmtNum(r.test_lift, 2), r.test_confirmed ? "✓" : "—"])} />} />
            </div>
            <div><b>{tx("5) مدة الإنذار المبكر", "5) Early-warning lead time")}:</b>{" "}
              <Ans v={s.q5_lead_time} render={(x) => x.map((r: any, i: number) => <Pill key={i}>{r.horizon}: {fmtNum(r.mean_warning_lead_sec, 1)}s · {tx("مكتشف", "detected")} {fmtPct(r.spikes_detected_share, 0)}</Pill>)} />
            </div>
            <div><b>{tx("6) نسبة الإنذارات الكاذبة", "6) False-alarm rate")}:</b>{" "}
              <Ans v={s.q6_false_alarms?.holdout} render={(x) => x.map((r: any, i: number) => <Pill key={i} tone="#f97316">{r.horizon}: FDR {fmtPct(r.false_discovery_rate, 1)} · FAR {fmtPct(r.false_alarm_rate, 1)}</Pill>)} />
            </div>
            <div><b>{tx("7) هل تستمر النتائج على البيانات الجديدة؟", "7) Do results hold on new data?")}</b>{" "}
              <Ans v={s.q7_forward} render={(x) => x.map((r: any, i: number) => <Pill key={i} tone={r.bss_vs_base_rate > 0 ? "#22c55e" : "#8b95a7"}>{r.horizon}: n={r.n} BSS {fmtNum(r.bss_vs_base_rate, 4)}</Pill>)} />
            </div>
            <details className="text-xs"><summary className="cursor-pointer text-muted">JSON</summary><Json data={s} /></details>
          </div>
        </Card>
      ))}
    </div>
  );
}

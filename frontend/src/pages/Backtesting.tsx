import { useEffect, useState } from "react";
import { api, fmtCI, fmtInt, fmtNum, fmtPct, useApi } from "../api";
import { LineSvg } from "../components/svg";
import { Btn, Card, Empty, Json, Loader, Pill, Select, Table, Verdict } from "../components/ui";
import { useSymbols } from "../hooks";
import { useI18n, useTx } from "../i18n";

export default function Backtesting() {
  const { t } = useI18n();
  const tx = useTx();
  const { symbols } = useSymbols();
  const [symbol, setSymbol] = useState("");
  const [hz, setHz] = useState("");
  useEffect(() => { if (!symbol && symbols.length) setSymbol(symbols[0].symbol); }, [symbols, symbol]);
  const { data, error, loading } = useApi<any>(symbol ? `/backtests?symbol=${symbol}` : null, 60000);
  const horizons = Object.keys(data?.horizons || {});
  useEffect(() => { if (horizons.length && !horizons.includes(hz)) setHz(horizons[0]); }, [horizons.join(), hz]);
  const g = data?.horizons?.[hz];
  const s = g?.summary;
  const acc = s?.acceptance;
  const champ = s?.selected;
  const results = g?.results || {};

  return (
    <div className="space-y-4">
      <Card title={t("nav_backtest")} right={
        <div className="flex flex-wrap gap-2">
          <Select label={t("symbol")} value={symbol} onChange={setSymbol} options={symbols.map((x) => ({ value: x.symbol, label: x.display_name }))} />
          {horizons.length > 0 && <Select label={t("horizon")} value={hz} onChange={setHz} options={horizons.map((x) => ({ value: x, label: x }))} />}
          <Btn onClick={() => api(`/jobs/train?symbol=${symbol}`, { method: "POST" })}>{t("run_now")}</Btn>
        </div>}>
        <p className="mb-3 text-xs text-muted">
          {tx("التقسيم الزمني: تطوير 60% (Walk-forward مُطهّر للاختيار) ← تحقق 20% (معايرة + حدود الإشارات) ← اختبار نهائي 20% (بوابة قبول فقط، لا يُستخدم للاختيار). تطهير بطول الأفق + حظر 500 تيك بين الكتل.",
              "Time split: DEV 60% (purged walk-forward for selection) → VAL 20% (calibration + signal cut-offs) → TEST 20% (acceptance gate only, never used for selection). Purge = horizon length, embargo = 500 ticks between blocks.")}
        </p>
        <Loader error={error} loading={loading && !data} />
        {!horizons.length && !loading && <Empty>{tx("لا توجد نتائج تدريب بعد", "No training results yet")}</Empty>}
        {s && (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2 text-xs">
              <Pill>{tx("الحالة", "Status")}: {s.status}</Pill>
              <Pill tone="#a78bfa">{tx("الصفوف", "Rows")} dev/val/test: {s.rows?.dev}/{s.rows?.val}/{s.rows?.test}</Pill>
              <Pill tone="#a78bfa">{tx("الإيجابيات", "Positives")}: {s.positives?.dev}/{s.positives?.val}/{s.positives?.test}</Pill>
              {champ && <Pill tone="#38bdf8">{tx("المختار", "Selected")}: {champ}</Pill>}
              {s.best_baseline && <Pill tone="#8b95a7">{tx("أفضل مرجعي (على التحقق)", "Best baseline (on val)")}: {s.best_baseline}</Pill>}
            </div>
            {(s.notes || []).map((n: string, i: number) => <div key={i} className="text-xs text-amber-300">{n}</div>)}
            {acc && (
              <>
                <Verdict ok={!!s.approved}>
                  {s.approved
                    ? tx("النموذج اجتاز جميع شروط القبول على بيانات الاختبار غير المرئية.", "Model passed every acceptance condition on unseen test data.")
                    : tx("النموذج غير معتمد: لم يتفوق بثقة على النماذج المرجعية على بيانات الاختبار.", "Model NOT approved: it did not reliably beat the reference baselines on test data.")}
                </Verdict>
                <Table dense head={[tx("الشرط", "Condition"), tx("المتوسط", "Mean"), "95% CI", tx("النتيجة", "Result")]}
                  rows={Object.entries<any>(acc).map(([k, v]) => [k, fmtNum(v.mean ?? v.value, 4), v.ci95 ? fmtCI(v.ci95[0], v.ci95[1]) : "—",
                    v.pass ? <span className="text-emerald-400">PASS</span> : <span className="text-red-400">FAIL</span>])} />
              </>
            )}
            <h3 className="text-sm font-semibold">{tx("المقارنة على كل الأقسام", "Comparison across splits")}</h3>
            <Table dense
              head={[t("model"), tx("النوع", "Kind"), "IS BSS", "WF BSS", "Val BSS", "Test BSS", "Test PR-AUC", "Test Brier", "Precision", "Recall", "F1", tx("إنذار كاذب", "False alarm"), "ECE"]}
              rows={Object.entries<any>(results).map(([name, r]) => {
                const tt = r.test || {};
                const at = tt.at_threshold || r.validation?.at_threshold || {};
                return [
                  <b className={name === champ ? "text-accent" : ""}>{name}</b>, r.kind,
                  fmtNum(r.in_sample?.bss_vs_base_rate, 4), fmtNum(r.walk_forward?.wf_bss_mean, 4), fmtNum(r.validation?.bss_vs_base_rate, 4),
                  fmtNum(tt.bss_vs_base_rate, 4), fmtNum(tt.pr_auc, 3), fmtNum(tt.brier, 5),
                  fmtPct(at.precision, 1), fmtPct(at.recall, 1), fmtNum(at.f1, 3), fmtPct(at.false_alarm_rate, 1), fmtNum((tt.calibration || r.validation?.calibration)?.ece, 4),
                ];
              })} />
            {champ && results[champ]?.test && <ChampionDetail t={results[champ].test} tx={tx} />}
            {s.thresholds && (
              <div>
                <h3 className="mb-1 text-sm font-semibold">{tx("حدود المستويات (من التحقق)", "Level thresholds (from validation)")}</h3>
                <div className="flex flex-wrap gap-2 text-xs">
                  {["yellow", "orange", "red"].map((l) => <Pill key={l} tone={{ yellow: "#eab308", orange: "#f97316", red: "#ef4444" }[l]}>{l}: {s.thresholds[l] == null ? "—" : fmtPct(s.thresholds[l], 2)}</Pill>)}
                  <Pill tone="#8b95a7">{t("base_rate")}: {fmtPct(s.thresholds.base_rate, 2)}</Pill>
                </div>
                <p className="mt-1 text-[11px] text-muted">{s.thresholds.rule}</p>
              </div>
            )}
            <details className="text-xs"><summary className="cursor-pointer text-muted">JSON</summary><Json data={s} /></details>
          </div>
        )}
      </Card>
    </div>
  );
}

function ChampionDetail({ t, tx }: { t: any; tx: (a: string, b: string) => string }) {
  const vt = t.virtual_trading;
  const bm = t.virtual_trading_benchmark;
  const cal = t.calibration?.bins || [];
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-2 text-xs">
        <h3 className="text-sm font-semibold">{tx("البطل على الاختبار", "Champion on test")}</h3>
        <div>{tx("الانفجارات المكتشفة مسبقًا", "Spikes detected in advance")}: {fmtPct(t.event_detection?.detected_share, 1)} ({t.event_detection?.detected}/{t.event_detection?.spikes}) · {tx("متوسط مدة الإنذار", "mean lead")}: {fmtNum(t.mean_warning_lead_sec, 1)}s</div>
        <div className="text-muted">{t.event_detection?.note}</div>
        {vt && (
          <Table dense head={["", tx("صفقات", "Trades"), "PF", "MaxDD%", tx("صافي%", "Net%"), tx("فوز", "Win")]}
            rows={[[tx("الإشارة", "Signal"), fmtInt(vt.trades), fmtNum(vt.profit_factor, 2), fmtNum(vt.max_drawdown_pct, 2), fmtNum(vt.net_return_pct, 2), fmtPct(vt.win_rate, 1)],
                   bm ? [tx("مرجعي", "Benchmark"), fmtInt(bm.trades), fmtNum(bm.profit_factor, 2), fmtNum(bm.max_drawdown_pct, 2), fmtNum(bm.net_return_pct, 2), fmtPct(bm.win_rate, 1)] : []]} />
        )}
        <div className="text-muted">{tx("تداول افتراضي فقط — لا تُفتح أي صفقات.", "Virtual trading only — no trades are ever opened.")}</div>
      </div>
      <div>
        <h3 className="mb-1 text-sm font-semibold">{tx("منحنى المعايرة (الاختبار)", "Reliability diagram (test)")} — ECE {fmtNum(t.calibration?.ece, 4)}</h3>
        {cal.length ? <LineSvg xs={cal.map((b: any) => b.p_mean)} series={[{ values: cal.map((b: any) => b.y_rate), color: "#38bdf8" }, { values: cal.map((b: any) => b.p_mean), color: "#64748b", dash: true }]} zeroLine={false} xLabel="predicted p" /> : <Empty />}
      </div>
    </div>
  );
}

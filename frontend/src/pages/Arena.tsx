import { useEffect, useState } from "react";
import { api, fmtInt, fmtNum, fmtPct, fmtTime, useApi } from "../api";
import { Btn, Card, Empty, Loader, Pill, Select, Table, Verdict } from "../components/ui";
import { useSymbols } from "../hooks";
import { useI18n, useTx } from "../i18n";

export const verdictText = (v: string | undefined, tx: (a: string, b: string) => string) =>
  !v ? "—" : v.startsWith("validated")
    ? tx("توجد قواعد اجتازت التحقق (FDR) وتأكدت على الاختبار النهائي", "Rules passed validation (FDR) and were confirmed on the final test")
    : tx("لم تنجُ أي قاعدة من تصحيح FDR على التحقق مع تأكيد على الاختبار — لا دليل على نمط قابل للاستغلال لهذا الأفق",
         "No rule survived validation FDR + holdout confirmation — no evidence of an exploitable pre-spike pattern for this horizon");

const pf = (x: any) => (x == null ? "—" : !isFinite(x) ? "∞" : fmtNum(x, 2));

export default function Arena() {
  const { t } = useI18n();
  const tx = useTx();
  const { symbols } = useSymbols();
  const [symbol, setSymbol] = useState("");
  const [hz, setHz] = useState("");
  useEffect(() => { if (!symbol && symbols.length) setSymbol(symbols[0].symbol); }, [symbols, symbol]);
  const { data, error, loading } = useApi<any>(symbol ? `/arena?symbol=${symbol}` : null, 60000);
  const a = data?.[symbol];
  const horizons = Object.keys(a?.horizons || {});
  useEffect(() => { if (horizons.length && !horizons.includes(hz)) setHz(horizons[0]); }, [horizons.join(), hz]);
  const h = a?.horizons?.[hz];
  const exps = useApi<any[]>(symbol && hz ? `/arena/experiments?symbol=${symbol}&horizon=${hz}&split=val&limit=60` : null, 0);

  return (
    <div className="space-y-4">
      <Card title={t("nav_arena")} right={
        <div className="flex flex-wrap gap-2">
          <Select label={t("symbol")} value={symbol} onChange={setSymbol} options={symbols.map((s) => ({ value: s.symbol, label: s.display_name }))} />
          {horizons.length > 0 && <Select label={t("horizon")} value={hz} onChange={setHz} options={horizons.map((x) => ({ value: x, label: x }))} />}
          <Btn onClick={() => api(`/jobs/arena?symbol=${symbol}`, { method: "POST" })}>{t("run_now")}</Btn>
        </div>}>
        <p className="mb-3 text-xs text-muted">
          {tx("كل قاعدة = (عائلة، فريم، مؤشر، اتجاه، كمية). الحدود تُحسب على بيانات التطوير فقط؛ الاختيار على التحقق مع تصحيح FDR؛ الاختبار النهائي للتقرير فقط. التقييم على محاولات غير متداخلة (إشارة واحدة لكل نافذة أفق) لتجنب الدلالة المضللة.",
              "Each rule = (family, timeframe, feature, side, quantile). Cut-offs from DEV only; selection on VALIDATION with FDR; TEST is report-only. Signals are scored as non-overlapping trials (one per horizon window) to avoid spurious significance.")}
        </p>
        <Loader error={error} loading={loading && !a} />
        {!a && !loading && <Empty>{tx("لم يُشغَّل المختبر لهذا المؤشر بعد", "Arena has not run for this symbol yet")}</Empty>}
        {a && a.status !== "ok" && <Empty>{a.status}</Empty>}
        {h?.status === "ok" && (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-3 text-xs">
              <Pill>{tx("قواعد مُختبرة", "Rules tested")}: {fmtInt(h.rules_tested)}</Pill>
              <Pill tone="#a78bfa">{tx("عائلات", "Families")}: {Object.keys(h.rules_per_family || {}).length}</Pill>
              <Pill tone="#8b95a7">{tx("اجتازت التصفية", "Screened on dev")}: {h.screened_on_dev}</Pill>
              <Pill tone="#eab308">{tx("دالة على التحقق (FDR 10%)", "Val-significant (FDR 10%)")}: {h.val_significant_fdr10}</Pill>
              <Pill tone="#22c55e">{tx("مؤكدة على الاختبار", "Test-confirmed")}: {h.test_confirmed}</Pill>
              <Pill tone="#8b95a7">{t("base_rate")} (test): {fmtPct(h.base_rate_test)}</Pill>
              <span className="text-muted">{fmtTime(a._created_at)} · {a.run_id}</span>
            </div>
            <Verdict ok={h.test_confirmed > 0}>{verdictText(h.verdict, tx)}</Verdict>
            <h3 className="text-sm font-semibold">{tx("أفضل قاعدة لكل عائلة (مختارة على التحقق)", "Best rule per family (chosen on validation)")}</h3>
            <Table dense
              head={[tx("العائلة", "Family"), t("timeframe"), tx("القاعدة", "Rule"), tx("رفع dev", "Dev lift"), tx("رفع val", "Val lift"), "q val", tx("رفع test", "Test lift"), "p test", tx("محاولات test", "Test trials"), "PF", "MaxDD%", tx("صافي%", "Net%"), tx("صفقات", "Trades"), ""]}
              rows={(h.families || []).map((f: any) => [
                f.family, f.timeframe, <span className="font-mono text-[11px]">{f.feature} {f.side} {f.q ?? ""}</span>,
                fmtNum(f.dev?.lift, 2), fmtNum(f.val?.lift, 2), fmtNum(f.val?.q_value, 3), fmtNum(f.test?.lift, 2), fmtNum(f.test?.p_value, 3), fmtInt(f.test?.n_signals),
                pf(f.virtual_trading_test?.profit_factor), fmtNum(f.virtual_trading_test?.max_drawdown_pct, 2), fmtNum(f.virtual_trading_test?.net_return_pct, 2), fmtInt(f.virtual_trading_test?.trades),
                f.val_significant ? (f.test?.confirmed ? <Pill tone="#22c55e">{tx("مؤكد", "confirmed")}</Pill> : <Pill tone="#eab308">{tx("لم يتأكد", "not confirmed")}</Pill>) : <Pill tone="#6b7280">n.s.</Pill>,
              ])} />
            <div className="text-xs text-muted">
              {tx("المعيار المرجعي للتداول الافتراضي (دخول في كل فرصة بلا إشارة)", "Virtual trading benchmark (enter at every opportunity, no signal)")}: PF {pf(h.trading_benchmark_test?.profit_factor)} · {tx("صافي", "net")} {fmtNum(h.trading_benchmark_test?.net_return_pct, 2)}% · {fmtInt(h.trading_benchmark_test?.trades)} {tx("صفقة", "trades")} — {tx("افتراضي فقط، تكلفة لكل صفقة", "virtual only, cost per trade")} {h.trading_benchmark_test?.cost_pct_per_trade}%
            </div>
            {h.best_combo && (
              <div>
                <h3 className="mb-1 text-sm font-semibold">{tx("أفضل دمج (AND)", "Best combination (AND)")}</h3>
                <Table dense head={[tx("الدمج", "Combo"), t("timeframe"), tx("رفع val", "Val lift"), "q", tx("رفع test", "Test lift"), "p test", "PF"]}
                  rows={[[<span className="font-mono text-[11px]">{h.best_combo.feature}</span>, h.best_combo.timeframe, fmtNum(h.best_combo.val?.lift, 2), fmtNum(h.best_combo.val?.q_value, 3),
                    fmtNum(h.best_combo.test?.lift, 2), fmtNum(h.best_combo.test?.p_value, 3), pf(h.best_combo.virtual_trading_test?.profit_factor)]]} />
              </div>
            )}
            <div>
              <h3 className="mb-1 text-sm font-semibold">{tx("أقوى القواعد على التحقق (الترتيب حسب p)", "Strongest rules on validation (by p)")}</h3>
              {exps.data?.length ? (
                <Table dense head={[tx("العائلة", "Family"), t("timeframe"), tx("القاعدة", "Rule"), tx("محاولات", "Trials"), "Precision", t("base_rate"), t("lift"), "p", "q"]}
                  rows={exps.data.map((e: any) => [e.family, e.timeframe, <span className="font-mono text-[11px]">{e.params?.feature} {e.params?.side} {e.params?.q ?? ""}</span>, e.n_signals,
                    fmtPct(e.precision, 1), fmtPct(e.base_rate, 1), fmtNum(e.lift, 2), fmtNum(e.p_value, 4), fmtNum(e.q_value, 3)])} />
              ) : <Empty />}
            </div>
          </div>
        )}
        {h && h.status !== "ok" && <Empty>{h.status}</Empty>}
      </Card>
    </div>
  );
}

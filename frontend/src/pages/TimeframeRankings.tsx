import { fmtNum, fmtTime, useApi } from "../api";
import { Card, Empty, Loader, Pill, Table } from "../components/ui";
import { verdictText } from "./Arena";
import { useI18n, useTx } from "../i18n";

export default function TimeframeRankings() {
  const { t } = useI18n();
  const tx = useTx();
  const { data, error, loading } = useApi<any>("/rankings/timeframes", 60000);
  const syms = Object.entries<any>(data || {});
  return (
    <div className="space-y-4">
      <Card title={t("nav_tf")}>
        <p className="mb-3 text-xs text-muted">
          {tx("الترتيب حسب عدد القواعد الدالة على بيانات التحقق وأفضل رفع، مع إظهار هل تأكدت أفضل قاعدة لكل فريم على الاختبار النهائي.",
              "Ranked by number of validation-significant rules and best lift; shows whether each timeframe's best rule was confirmed on the final test.")}
        </p>
        <Loader error={error} loading={loading && !data} />
        {data && !syms.length && <Empty />}
      </Card>
      {syms.map(([sym, s]) => (
        <Card key={sym} title={s.display_name} right={<span className="text-xs text-muted">{fmtTime(s._created_at)}</span>}>
          <div className="grid gap-4 xl:grid-cols-2">
            {Object.entries<any>(s.horizons || {}).map(([h, v]) => (
              <div key={h}>
                <div className="mb-1 flex items-center gap-2 text-sm"><b>{t("horizon")} {h}</b><span className="text-xs text-muted">{verdictText(v.verdict, tx)}</span></div>
                {v.ranking?.length ? (
                  <Table dense head={["#", t("timeframe"), tx("مُصفّاة", "Screened"), tx("دالة", "Val-sig"), tx("أفضل رفع val", "Best val lift"), tx("أفضل قاعدة", "Best rule"), tx("رفع test", "Test lift"), ""]}
                    rows={v.ranking.map((r: any, i: number) => [i + 1, <b>{r.timeframe}</b>, r.screened, r.val_significant, fmtNum(r.best_val_lift, 2),
                      <span className="font-mono text-[11px]">{r.best_rule || "—"}</span>, fmtNum(r.test_lift, 2),
                      r.test_confirmed ? <Pill tone="#22c55e">✓ test</Pill> : r.val_significant ? <Pill tone="#eab308">val only</Pill> : <Pill tone="#6b7280">—</Pill>])} />
                ) : <Empty>{tx("لا توجد قواعد اجتازت التصفية", "No rules passed screening")}</Empty>}
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  );
}

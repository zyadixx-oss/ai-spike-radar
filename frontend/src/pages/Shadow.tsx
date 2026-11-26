import { fmtInt, fmtNum, fmtPct, ltr, useApi } from "../api";
import { Card, Empty, Loader, Table } from "../components/ui";
import { useI18n, useTx } from "../i18n";

function StatsTable({ rows }: { rows: any[] }) {
  const { t } = useI18n();
  const tx = useTx();
  if (!rows?.length) return <Empty>{tx("لا توجد توقعات مقيّمة بعد", "No evaluated predictions yet")}</Empty>;
  return (
    <Table dense
      head={[t("symbol"), t("horizon"), t("model"), "N", tx("تحقق", "Hits"), tx("المعدل الفعلي", "Observed"), tx("متوسط p", "Mean p"), "Brier", "BSS", "PR-AUC",
        tx("إنذارات", "Alerts"), tx("دقة الإنذار", "Alert precision"), "Recall", tx("إنذار كاذب", "False alarm"), tx("مدة الإنذار", "Lead"), "ECE"]}
      rows={rows.map((r) => [r.symbol, r.horizon, r.model_name, fmtInt(r.n), fmtInt(r.hits), fmtPct(r.observed_rate), fmtPct(r.mean_probability), fmtNum(r.brier, 5),
        <span className={r.bss_vs_base_rate > 0 ? "text-emerald-400" : "text-muted"}>{fmtNum(r.bss_vs_base_rate, 4)}</span>, fmtNum(r.pr_auc, 3),
        fmtInt(r.alerts), r.alert_precision_ci ? ltr(`${fmtPct(r.alert_precision, 1)} [${fmtPct(r.alert_precision_ci[0], 0)}–${fmtPct(r.alert_precision_ci[1], 0)}]`) : "—",
        fmtPct(r.recall, 1), fmtPct(r.false_alarm_rate, 1), r.mean_lead_sec != null ? `${fmtNum(r.mean_lead_sec, 0)}s` : "—", fmtNum(r.calibration?.ece, 4)])} />
  );
}

export default function Shadow() {
  const { t } = useI18n();
  const tx = useTx();
  const { data, error, loading } = useApi<any>("/shadow", 30000);
  return (
    <div className="space-y-4">
      <Card title={t("nav_shadow")}>
        <p className="text-xs text-muted">
          {tx("الاختبار الأمامي: أداء التوقعات الحية بعد انتهاء صلاحيتها على بيانات لم تكن موجودة وقت التدريب. نماذج الظل (المتحدّون) تعمل بالتوازي ولا تظهر كإشارات؛ تُستبدل النماذج فقط عند إثبات تحسن خارج بيانات التدريب.",
              "Forward testing: live predictions scored after expiry on data that did not exist at training time. Shadow (challenger) models run in parallel and are never shown as signals; a champion is replaced only on proven out-of-sample improvement.")}
        </p>
        <Loader error={error} loading={loading && !data} />
      </Card>
      <Card title={tx("الأبطال (اختبار أمامي)", "Champions (forward test)")}><StatsTable rows={data?.champion} /></Card>
      <Card title={tx("نماذج الظل", "Shadow challengers")}><StatsTable rows={data?.shadow} /></Card>
      <Card title={tx("المرجع: المعدل الطبيعي عديم الذاكرة", "Reference: memoryless base rate")}><StatsTable rows={data?.baseline} /></Card>
    </div>
  );
}

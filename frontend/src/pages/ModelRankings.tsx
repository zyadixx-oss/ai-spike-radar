import { fmtNum, fmtTime, useApi } from "../api";
import { Card, Empty, Loader, Pill, Table } from "../components/ui";
import { useI18n, useTx } from "../i18n";

export default function ModelRankings() {
  const { t } = useI18n();
  const tx = useTx();
  const { data, error, loading } = useApi<any>("/rankings/models", 60000);
  const rk = Object.entries<any>(data?.rankings || {});
  const reg = data?.registry || [];
  return (
    <div className="space-y-4">
      <Card title={t("nav_models")}>
        <p className="mb-3 text-xs text-muted">
          {tx("الترتيب حسب متوسط Brier Skill Score مقابل المعدل الطبيعي في Walk-forward خارج العينة. النماذج المرجعية (عشوائي، معدل أساسي، هندسي، فواصل، Weibull) تُعرض للمقارنة. عند التعادل يُفضَّل النموذج الأبسط.",
              "Ranked by mean out-of-sample walk-forward Brier Skill Score vs the base rate. Reference baselines (random, base rate, geometric, interval-hazard, Weibull) shown for comparison. Ties favour the simpler model.")}
        </p>
        <Loader error={error} loading={loading && !data} />
        {data && !rk.length && <Empty>{tx("لا توجد نتائج تدريب بعد", "No training results yet")}</Empty>}
      </Card>
      {reg.length > 0 && (
        <Card title={tx("سجل النماذج (بطل / متحدٍّ)", "Model registry (champion / challenger)")}>
          <Table dense head={["ID", t("symbol"), t("horizon"), t("model"), tx("الدور", "Role"), t("approved"), tx("أنشئ", "Created"), tx("ملاحظة", "Note")]}
            rows={reg.map((m: any) => [m.id, m.symbol, m.horizon, m.model_name, m.role === "champion" ? <Pill>champion</Pill> : <Pill tone="#a78bfa">challenger</Pill>,
              m.approved ? <span className="text-emerald-400">✓</span> : "—", <span className="font-mono text-xs">{fmtTime(m.created_at)}</span>, <span className="text-xs text-muted">{m.notes}</span>])} />
        </Card>
      )}
      {rk.map(([sym, hz]) => (
        <Card key={sym} title={sym}>
          <div className="grid gap-4 xl:grid-cols-2">
            {Object.entries<any>(hz).map(([h, g]) => (
              <div key={h}>
                <div className="mb-1 flex flex-wrap items-center gap-2 text-sm">
                  <b>{h}</b>
                  {g.summary?.selected && <Pill>{tx("مختار", "selected")}: {g.summary.selected}</Pill>}
                  {g.summary?.approved ? <Pill tone="#22c55e">{t("approved")}</Pill> : <Pill tone="#f59e0b">{t("not_approved")}</Pill>}
                  <span className="text-xs text-muted">{g.summary?.status}</span>
                </div>
                <Table dense head={["#", t("model"), tx("نوع", "Kind"), "WF BSS", "WF PR-AUC", "WF Brier", tx("طيات", "Folds"), tx("ثوانٍ", "Secs"), ""]}
                  rows={g.models.map((m: any, i: number) => [i + 1, <b className={m.name === g.summary?.selected ? "text-accent" : ""}>{m.name}</b>,
                    m.kind === "baseline" ? <Pill tone="#8b95a7">baseline</Pill> : "model", fmtNum(m.wf_bss_mean, 4), fmtNum(m.wf_pr_auc_mean, 3), fmtNum(m.wf_brier_mean, 5),
                    m.folds?.length ?? "—", fmtNum(m.fit_seconds, 1), m.status !== "ok" ? <span className="text-xs text-amber-300" title={m.reason}>{m.status}: {String(m.reason || "").slice(0, 60)}</span> : null])} />
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  );
}

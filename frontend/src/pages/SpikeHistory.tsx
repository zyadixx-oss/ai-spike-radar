import { useEffect, useState } from "react";
import { api, fmtCI, fmtInt, fmtNum, fmtTime, ltr, useApi } from "../api";
import { BarsSvg, LineSvg } from "../components/svg";
import { Btn, Card, Empty, Loader, Pill, Select, Stat, Table, Verdict } from "../components/ui";
import { useSymbols } from "../hooks";
import { useI18n, useTx } from "../i18n";

export default function SpikeHistory() {
  const { t } = useI18n();
  const tx = useTx();
  const { symbols } = useSymbols();
  const [symbol, setSymbol] = useState("");
  const [cls, setCls] = useState("");
  const [page, setPage] = useState(0);
  useEffect(() => { if (!symbol && symbols.length) setSymbol(symbols[0].symbol); }, [symbols, symbol]);
  const res = useApi<any>(symbol ? `/research/${symbol}` : null, 60000);
  const list = useApi<any>(symbol ? `/spikes?symbol=${symbol}&limit=100&offset=${page * 100}${cls ? `&size_class=${cls}` : ""}` : null, 15000);
  const r = res.data?.research;
  const ia = r?.interarrival;
  const st = r?.spike_stats;
  const pt = r?.patterns;
  const es = r?.event_study;

  return (
    <div className="space-y-4">
      <Card title={t("nav_spikes")} right={
        <div className="flex flex-wrap gap-2">
          <Select label={t("symbol")} value={symbol} onChange={(v) => { setSymbol(v); setPage(0); }} options={symbols.map((s) => ({ value: s.symbol, label: s.display_name }))} />
          <Btn onClick={() => api(`/jobs/research?symbol=${symbol}`, { method: "POST" })}>{t("run_now")}</Btn>
        </div>}>
        <Loader error={res.error} loading={res.loading && !r} />
        {!r && !res.loading && <Empty>{tx("لم يكتمل التحليل البحثي لهذا المؤشر بعد", "Research has not run for this symbol yet")}</Empty>}
        {r && r.status !== "ok" && <Empty>{r.status}: {fmtInt(r.ticks)} ticks</Empty>}
        {r?.status === "ok" && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
              <Stat label={t("spikes")} value={fmtInt(st?.count)} sub={ltr(`${fmtNum(st?.per_1000_ticks, 2)} / 1000 ticks`)} />
              <Stat label={tx("متوسط الفاصل", "Mean interval")} value={ltr(`${fmtNum(ia?.mean_ticks, 0)} ticks`)} sub={ltr(`${fmtNum(ia?.mean_seconds, 0)} s`)} />
              <Stat label={tx("شكل Weibull", "Weibull shape")} value={fmtNum(ia?.weibull_shape, 3)} sub={<>95% CI {fmtCI(ia?.weibull_shape_ci95?.[0], ia?.weibull_shape_ci95?.[1], 2)}</>} />
              <Stat label="KS p (exp)" value={fmtNum(ia?.ks_vs_exponential?.p_value, 3)} />
              <Stat label={tx("تغطية البيانات", "Coverage")} value={ltr(`${fmtNum(r.coverage?.days, 1)} d`)} sub={`${r.coverage?.segments} ${tx("مقاطع", "segments")}`} />
              <Stat label={tx("تعريف الانفجار", "Spike definition")} value={`k=${r.definition?.k}`} sub={ltr(`σ=${fmtNum(r.definition?.sigma * 1e4, 2)}e-4`)} />
            </div>
            <Verdict ok={ia?.consistent_with_memoryless === false}>
              {ia?.consistent_with_memoryless
                ? tx("توقيت الانفجارات متسق مع عملية عديمة الذاكرة: تأخر الانفجار لا يعني اقترابه. عدد التيكات منذ آخر انفجار لا يحمل قيمة تنبؤية بحسب هذه البيانات.",
                     "Spike timing is consistent with a memoryless process: a 'late' spike is NOT more likely. Ticks since the last spike carry no predictive value in this data.")
                : tx("توقيت الانفجارات يختلف عن العملية عديمة الذاكرة — راجع منحنى Hazard.", "Spike timing deviates from a memoryless process — see the hazard curve.")}
            </Verdict>
            <div className="grid gap-4 lg:grid-cols-2">
              <div>
                <h3 className="mb-1 text-xs text-muted">{tx("معدل الخطر (Hazard) لكل تيك حسب عدد التيكات منذ آخر انفجار — الخط البرتقالي = المعدل الثابت", "Hazard per tick vs ticks since last spike — orange = constant (memoryless) rate")}</h3>
                <BarsSvg values={(ia?.hazard_bins || []).map((h: any) => h.hazard_per_tick || 0)} labels={(ia?.hazard_bins || []).map((h: any) => String(Math.round(h.to)))} refLine={ia?.mean_ticks ? 1 / ia.mean_ticks : undefined} />
              </div>
              <div>
                <h3 className="mb-1 text-xs text-muted">{tx("دراسة الحدث: متوسط المسار حول الانفجار (σ)", "Event study: mean path around spikes (σ units)")}</h3>
                {es?.status === "ok" ? (
                  <LineSvg xs={es.offsets} series={[{ values: es.mean, color: "#38bdf8" }, { values: es.p25, color: "#64748b", dash: true }, { values: es.p75, color: "#64748b", dash: true }]} xLabel="ticks" />
                ) : <Empty />}
              </div>
              <div>
                <h3 className="mb-1 text-xs text-muted">{tx("الانفجارات حسب الساعة (UTC)", "Spikes by hour (UTC)")} — χ² p={fmtNum(st?.hour_uniformity_chi2?.p_value, 3)}</h3>
                <BarsSvg values={st?.by_hour_utc || []} labels={(st?.by_hour_utc || []).map((_: any, i: number) => String(i))} color="#a78bfa" />
              </div>
              <div>
                <h3 className="mb-1 text-xs text-muted">{tx("حساسية العدد لقيمة k", "Count sensitivity to k")}</h3>
                <Table dense head={["k", t("spikes"), "/1000"]} rows={(r.threshold_sensitivity || []).map((x: any) => [x.k, fmtInt(x.count), fmtNum(x.per_1000_ticks, 2)])} />
                <div className="mt-2 text-xs text-muted">{tx("التصنيف", "Classes")}: {Object.entries(st?.size_classes || {}).map(([k, v]) => `${k}=${v}`).join(" · ")}</div>
              </div>
            </div>
            <div>
              <h3 className="mb-2 text-sm font-semibold">{tx("ما قبل الانفجار مقابل نوافذ ضابطة", "Pre-spike windows vs control windows")}
                <span className="ms-2 text-xs font-normal text-muted">{pt?.events} {tx("حدث", "events")} · {pt?.controls} {tx("ضابطة", "controls")} · {tx("قوية", "robust")}: {pt?.n_robust} / {pt?.tests?.length}</span>
              </h3>
              <p className="mb-2 text-xs text-muted">{tx("قوية = q<0.05 بعد تصحيح FDR وبنفس الاتجاه مع p<0.05 في نصفي التاريخ كليهما. AUC = 0.5 يعني لا فرق بين نوافذ ما قبل الانفجار والنوافذ الضابطة.",
                "robust = FDR q<0.05 AND same-direction p<0.05 in both halves of history. AUC 0.5 = no difference between pre-spike and control windows.")}</p>
              {pt?.tests?.length ? (
                <Table dense head={[tx("النافذة", "Window"), tx("المقياس", "Stat"), tx("وسيط قبل", "Pre median"), tx("وسيط ضابط", "Ctrl median"), "AUC", "p", "q (FDR)", tx("تكرار", "Replicated"), ""]}
                  rows={pt.tests.slice(0, 40).map((x: any) => [x.window, x.stat, fmtNum(x.pre_median, 3), fmtNum(x.ctrl_median, 3), fmtNum(x.auc, 3), fmtNum(x.p_value, 4), fmtNum(x.q_value, 4),
                    x.replicated ? "✓" : "—", x.robust ? <Pill tone="#22c55e">robust</Pill> : x.significant ? <Pill tone="#eab308">FDR only</Pill> : null])} />
              ) : <Empty />}
            </div>
          </div>
        )}
      </Card>
      <Card title={tx("قائمة الانفجارات", "Spike list")} right={
        <div className="flex gap-2">
          <Select value={cls} onChange={(v) => { setCls(v); setPage(0); }} options={[{ value: "", label: t("all") }, { value: "small", label: "small" }, { value: "medium", label: "medium" }, { value: "large", label: "large" }]} />
          <Btn disabled={page === 0} onClick={() => setPage(page - 1)}>‹</Btn>
          <span className="self-center text-xs text-muted">{page + 1} / {Math.max(1, Math.ceil((list.data?.total || 0) / 100))}</span>
          <Btn disabled={(page + 1) * 100 >= (list.data?.total || 0)} onClick={() => setPage(page + 1)}>›</Btn>
        </div>}>
        {list.data?.rows?.length ? (
          <Table dense head={[t("time"), t("direction"), t("size"), "σ×", tx("الحركة", "Move"), tx("قبل", "Before"), tx("بعد", "After"), tx("تيكات منذ السابق", "Ticks since prev"), tx("ثوانٍ", "Secs")]}
            rows={list.data.rows.map((s: any) => [<span className="font-mono text-xs">{fmtTime(s.epoch)}</span>, s.direction > 0 ? "▲" : "▼", s.size_class, fmtNum(s.magnitude_sigma, 0),
              fmtNum(s.abs_move, 3), fmtNum(s.price_before, 3), fmtNum(s.price_after, 3), fmtInt(s.ticks_since_prev), fmtInt(s.secs_since_prev)])} />
        ) : <Empty />}
      </Card>
    </div>
  );
}

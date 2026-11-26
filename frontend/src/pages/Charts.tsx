import { useEffect, useMemo, useRef, useState } from "react";
import {
  ColorType, CrosshairMode, createChart, type IChartApi, type ISeriesApi, type SeriesMarker, type Time, type UTCTimestamp,
} from "lightweight-charts";
import { api, fmtPct, fmtTime, useLive } from "../api";
import { Btn, Card, Empty, levelColor, Pill, Select } from "../components/ui";
import { HORIZONS, TF_SECONDS, TIMEFRAMES, useSymbols } from "../hooks";
import { useI18n, useTx } from "../i18n";

type Bar = { time: number; open: number; high: number; low: number; close: number; ticks?: number | null };

function ema(vals: number[], n: number) {
  const k = 2 / (n + 1);
  const out: (number | null)[] = [];
  let e: number | null = null;
  vals.forEach((v, i) => {
    e = e === null ? v : v * k + e * (1 - k);
    out.push(i >= n - 1 ? e : null);
  });
  return out;
}
function bollinger(vals: number[], n = 20, m = 2) {
  const up: (number | null)[] = [], lo: (number | null)[] = [], mid: (number | null)[] = [];
  for (let i = 0; i < vals.length; i++) {
    if (i < n - 1) { up.push(null); lo.push(null); mid.push(null); continue; }
    const w = vals.slice(i - n + 1, i + 1);
    const mu = w.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(w.reduce((a, b) => a + (b - mu) ** 2, 0) / n);
    up.push(mu + m * sd); lo.push(mu - m * sd); mid.push(mu);
  }
  return { up, lo, mid };
}
function rsi(vals: number[], n = 14) {
  const out: (number | null)[] = [null];
  let ag = 0, al = 0;
  for (let i = 1; i < vals.length; i++) {
    const d = vals[i] - vals[i - 1];
    const g = Math.max(d, 0), l = Math.max(-d, 0);
    if (i <= n) { ag += g / n; al += l / n; out.push(i === n ? 100 - 100 / (1 + ag / (al || 1e-12)) : null); }
    else { ag = (ag * (n - 1) + g) / n; al = (al * (n - 1) + l) / n; out.push(100 - 100 / (1 + ag / (al || 1e-12))); }
  }
  return out;
}
const line = (bars: Bar[], v: (number | null)[]) =>
  bars.map((b, i) => (v[i] == null ? null : { time: b.time as UTCTimestamp, value: v[i] as number })).filter(Boolean) as any[];

export default function Charts() {
  const { t } = useI18n();
  const tx = useTx();
  const { symbols } = useSymbols();
  const [symbol, setSymbol] = useState("");
  const [tf, setTf] = useState("M1");
  const [horizon, setHorizon] = useState("");
  const [bars, setBars] = useState<Bar[]>([]);
  const [source, setSource] = useState("");
  const [markers, setMarkers] = useState<{ spikes: any[]; predictions: any[] }>({ spikes: [], predictions: [] });
  const [ind, setInd] = useState({ ema20: true, ema50: true, bb: false, rsi: true, prob: true });
  const [replay, setReplay] = useState<{ on: boolean; idx: number; playing: boolean; speed: number; picking: boolean }>({ on: false, idx: 0, playing: false, speed: 300, picking: false });
  const el = useRef<HTMLDivElement>(null);
  const elRsi = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const chartRsi = useRef<IChartApi | null>(null);
  const series = useRef<Record<string, ISeriesApi<any>>>({});
  const barsRef = useRef<Bar[]>([]);
  barsRef.current = bars;
  const replayRef = useRef(replay);
  replayRef.current = replay;

  useEffect(() => { if (!symbol && symbols.length) setSymbol(symbols[0].symbol); }, [symbols, symbol]);

  // create charts once
  useEffect(() => {
    if (!el.current || !elRsi.current) return;
    const opts = {
      layout: { background: { type: ColorType.Solid, color: "#0b0f17" }, textColor: "#9ca3af", fontSize: 11 },
      grid: { vertLines: { color: "#141b26" }, horzLines: { color: "#141b26" } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#1f2937" },
      timeScale: { borderColor: "#1f2937", timeVisible: true, secondsVisible: true },
      autoSize: true,
    } as const;
    const c = createChart(el.current, opts);
    const r = createChart(elRsi.current, { ...opts, timeScale: { ...opts.timeScale, visible: false } });
    chart.current = c;
    chartRsi.current = r;
    series.current.candle = c.addCandlestickSeries({ upColor: "#22c55e", downColor: "#ef4444", borderVisible: false, wickUpColor: "#22c55e", wickDownColor: "#ef4444" });
    series.current.ema20 = c.addLineSeries({ color: "#38bdf8", lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    series.current.ema50 = c.addLineSeries({ color: "#a78bfa", lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    series.current.bbu = c.addLineSeries({ color: "#64748b", lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
    series.current.bbl = c.addLineSeries({ color: "#64748b", lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
    // overlay scale (documented pattern: empty priceScaleId) so probabilities never stretch the price axis
    series.current.prob = c.addHistogramSeries({ priceScaleId: "", priceFormat: { type: "percent" }, lastValueVisible: false, priceLineVisible: false });
    series.current.prob.priceScale().applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
    series.current.rsi = r.addLineSeries({ color: "#f59e0b", lineWidth: 1, priceLineVisible: false });
    series.current.rsi.createPriceLine({ price: 70, color: "#374151", lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: "" });
    series.current.rsi.createPriceLine({ price: 30, color: "#374151", lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: "" });
    c.timeScale().subscribeVisibleLogicalRangeChange((rg) => { if (rg) r.timeScale().setVisibleLogicalRange(rg); });
    c.subscribeClick((p) => {
      const rp = replayRef.current;
      if (rp.picking && p.time) {
        const i = barsRef.current.findIndex((b) => b.time >= (p.time as number));
        if (i > 0) setReplay({ ...rp, on: true, picking: false, idx: i, playing: false });
      }
    });
    return () => { c.remove(); r.remove(); chart.current = null; };
  }, []);

  // load data
  useEffect(() => {
    if (!symbol) return;
    let cancel = false;
    api<any>(`/candles?symbol=${symbol}&tf=${tf}&limit=${tf === "tick" ? 3000 : 1500}`).then(async (d) => {
      if (cancel) return;
      setBars(d.candles);
      setSource(d.source);
      setReplay((r) => ({ ...r, on: false, playing: false }));
      if (d.candles.length) {
        const m = await api<any>(`/markers?symbol=${symbol}&start=${d.candles[0].time}&end=${d.candles[d.candles.length - 1].time + (TF_SECONDS[tf] || 1) + 3600}${horizon ? `&horizon=${horizon}` : ""}`);
        if (!cancel) setMarkers(m);
      } else setMarkers({ spikes: [], predictions: [] });
      chart.current?.timeScale().fitContent();
    }).catch(() => setBars([]));
    return () => { cancel = true; };
  }, [symbol, tf, horizon]);

  // live ticks
  const connected = useLive((m) => {
    if (m.type !== "tick" || m.symbol !== symbol || replayRef.current.on) return;
    const tfs = TF_SECONDS[tf];
    setBars((prev) => {
      if (!prev.length) return prev;
      const last = prev[prev.length - 1];
      if (tfs === 0) {
        if (m.epoch <= last.time) return prev;
        return [...prev.slice(-4999), { time: m.epoch, open: last.close, high: Math.max(last.close, m.quote), low: Math.min(last.close, m.quote), close: m.quote }];
      }
      const b = Math.floor(m.epoch / tfs) * tfs;
      if (b < last.time) return prev;
      if (b === last.time) {
        const u = { ...last, high: Math.max(last.high, m.quote), low: Math.min(last.low, m.quote), close: m.quote };
        return [...prev.slice(0, -1), u];
      }
      return [...prev, { time: b, open: m.quote, high: m.quote, low: m.quote, close: m.quote }];
    });
  }, symbol || undefined);

  const visible = replay.on ? bars.slice(0, replay.idx) : bars;

  const snap = useMemo(() => {
    const times = visible.map((b) => b.time);
    return (e: number) => {
      let lo = 0, hi = times.length - 1;
      if (hi < 0 || e < times[0]) return null;
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (times[mid] <= e) lo = mid; else hi = mid - 1; }
      return times[lo];
    };
  }, [visible]);

  // render
  useEffect(() => {
    const s = series.current;
    if (!s.candle) return;
    s.candle.setData(visible.map((b) => ({ ...b, time: b.time as UTCTimestamp })));
    const closes = visible.map((b) => b.close);
    s.ema20.setData(ind.ema20 ? line(visible, ema(closes, 20)) : []);
    s.ema50.setData(ind.ema50 ? line(visible, ema(closes, 50)) : []);
    const bb = bollinger(closes);
    s.bbu.setData(ind.bb ? line(visible, bb.up) : []);
    s.bbl.setData(ind.bb ? line(visible, bb.lo) : []);
    s.rsi.setData(ind.rsi ? line(visible, rsi(closes)) : []);
    const lastT = visible.length ? visible[visible.length - 1].time + (TF_SECONDS[tf] || 1) : 0;
    const mk: SeriesMarker<Time>[] = [];
    for (const sp of markers.spikes) {
      if (sp.epoch > lastT) continue;
      const tt = snap(sp.epoch);
      if (tt == null) continue;
      mk.push({ time: tt as UTCTimestamp, position: sp.direction > 0 ? "aboveBar" : "belowBar", color: sp.size_class === "large" ? "#facc15" : "#a16207",
        shape: sp.direction > 0 ? "arrowUp" : "arrowDown", size: sp.size_class === "large" ? 1 : 0.6, text: sp.size_class === "large" ? "L" : "" });
    }
    const probBars = new Map<number, { value: number; color: string }>();
    for (const p of markers.predictions) {
      if (p.issued_epoch > lastT) continue;
      const tt = snap(p.issued_epoch);
      if (tt == null) continue;
      // Outcome is only revealed in replay once the bar after expiry is visible (no hindsight in replay)
      const resolved = p.status && (!replay.on || (p.expires_epoch ?? p.issued_epoch) < lastT);
      const tag = resolved ? (p.status === "hit" ? "✓" : p.status === "miss" ? "✗" : "?") : "";
      mk.push({ time: tt as UTCTimestamp, position: "belowBar", color: levelColor[p.level], shape: "circle", text: `${tag}${Math.round(p.probability * 100)}%` });
      const prev = probBars.get(tt);
      if (!prev || prev.value < p.probability) probBars.set(tt, { value: p.probability, color: levelColor[p.level] });
    }
    mk.sort((a, b) => (a.time as number) - (b.time as number));
    s.candle.setMarkers(mk);
    s.prob.setData(ind.prob ? [...probBars.entries()].sort((a, b) => a[0] - b[0]).map(([time, v]) => ({ time: time as UTCTimestamp, ...v })) : []);
  }, [visible, markers, ind, snap, tf, replay.on]);

  // replay playback
  useEffect(() => {
    if (!replay.on || !replay.playing) return;
    const id = setInterval(() => setReplay((r) => (r.idx >= bars.length ? { ...r, playing: false } : { ...r, idx: r.idx + 1 })), replay.speed);
    return () => clearInterval(id);
  }, [replay.on, replay.playing, replay.speed, bars.length]);

  const stats = useMemo(() => {
    const ps = markers.predictions;
    const hit = ps.filter((p) => p.status === "hit").length, miss = ps.filter((p) => p.status === "miss").length;
    return { n: ps.length, hit, miss, spikes: markers.spikes.length };
  }, [markers]);

  return (
    <div className="space-y-3">
      <Card
        title={<span className="flex items-center gap-2">{t("nav_charts")} <Pill tone={connected ? "#22c55e" : "#ef4444"}>{connected ? t("live") : t("disconnected")}</Pill> {source && <Pill tone="#8b95a7">{source}</Pill>}</span>}
        right={
          <div className="flex flex-wrap items-center gap-2">
            <Select label={t("symbol")} value={symbol} onChange={setSymbol} options={symbols.map((s) => ({ value: s.symbol, label: s.display_name }))} />
            <Select label={t("horizon")} value={horizon} onChange={setHorizon} options={[{ value: "", label: t("all") }, ...HORIZONS.map((h) => ({ value: h, label: h }))]} />
          </div>
        }
      >
        <div className="mb-2 flex flex-wrap gap-1">
          {TIMEFRAMES.map((x) => <Btn key={x} active={tf === x} onClick={() => setTf(x)}>{x === "tick" ? "Tick" : x}</Btn>)}
        </div>
        <div className="mb-2 flex flex-wrap items-center gap-1 text-xs">
          <span className="text-muted">{t("indicators")}:</span>
          {(["ema20", "ema50", "bb", "rsi", "prob"] as const).map((k) => (
            <Btn key={k} active={ind[k]} onClick={() => setInd({ ...ind, [k]: !ind[k] })}>
              {{ ema20: "EMA 20", ema50: "EMA 50", bb: "Bollinger 20,2", rsi: "RSI 14", prob: tx("شريط الاحتمال", "Probability strip") }[k]}
            </Btn>
          ))}
        </div>
        {!symbol && <Empty>{tx("لم تُكتشف مؤشرات بعد", "No symbols discovered yet")}</Empty>}
        <div className="relative">
          <div ref={el} className="h-[52vh] min-h-[320px] w-full" />
          {symbol && !bars.length && <div className="absolute inset-0 flex items-center justify-center text-sm text-muted">{t("no_data")}</div>}
        </div>
        <div ref={elRsi} className={`mt-1 w-full ${ind.rsi ? "h-28" : "h-0"} overflow-hidden`} />
        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-line pt-3 text-xs">
          <b>{t("replay")}:</b>
          {!replay.on ? (
            <>
              <Btn active={replay.picking} onClick={() => setReplay({ ...replay, picking: !replay.picking })}>{tx("اختر نقطة البداية على الشارت", "Click chart to pick start")}</Btn>
              <Btn disabled={bars.length < 50} onClick={() => setReplay({ ...replay, on: true, idx: Math.floor(bars.length * 0.6), playing: false })}>{tx("ابدأ من 60%", "Start at 60%")}</Btn>
            </>
          ) : (
            <>
              <Btn onClick={() => setReplay({ ...replay, playing: !replay.playing })}>{replay.playing ? t("pause") : t("play")}</Btn>
              <Btn onClick={() => setReplay({ ...replay, idx: Math.max(1, replay.idx - 1), playing: false })}>◀ {t("step")}</Btn>
              <Btn onClick={() => setReplay({ ...replay, idx: Math.min(bars.length, replay.idx + 1), playing: false })}>{t("step")} ▶</Btn>
              <Select value={String(replay.speed)} onChange={(v) => setReplay({ ...replay, speed: Number(v) })}
                options={[{ value: "1000", label: "1×" }, { value: "300", label: "3×" }, { value: "100", label: "10×" }, { value: "30", label: "30×" }]} />
              <span className="font-mono text-muted">{replay.idx}/{bars.length} · {fmtTime(bars[replay.idx - 1]?.time)}</span>
              <Btn onClick={() => setReplay({ ...replay, on: false, playing: false })}>{t("exit_replay")}</Btn>
            </>
          )}
        </div>
        <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted">
          <span>⚡ {t("spikes")}: {stats.spikes}</span>
          <span>● {tx("إشارات (أصفر فأعلى)", "Signals (yellow+)")}: {stats.n}</span>
          <span className="text-emerald-400">✓ {stats.hit}</span>
          <span className="text-red-400">✗ {stats.miss}</span>
          <span>{tx("دقة الإشارات الظاهرة", "Visible signal precision")}: {fmtPct(stats.hit + stats.miss ? stats.hit / (stats.hit + stats.miss) : null, 1)}</span>
        </div>
      </Card>
    </div>
  );
}

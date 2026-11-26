/** Minimal inline SVG charts (no extra deps). */
export function LineSvg({ xs, series, height = 160, zeroLine = true, xLabel }: {
  xs: number[]; series: { values: number[]; color: string; dash?: boolean; label?: string }[]; height?: number; zeroLine?: boolean; xLabel?: string;
}) {
  const all = series.flatMap((s) => s.values).filter((v) => isFinite(v));
  if (!xs.length || !all.length) return null;
  const W = 600, H = height, P = 28;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...all, zeroLine ? 0 : Infinity), y1 = Math.max(...all, zeroLine ? 0 : -Infinity);
  const sx = (x: number) => P + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * P);
  const sy = (y: number) => H - P + (-(y - y0) / (y1 - y0 || 1)) * (H - 2 * P);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ direction: "ltr" }}>
      {zeroLine && <line x1={P} x2={W - P} y1={sy(0)} y2={sy(0)} stroke="#374151" strokeDasharray="3 3" />}
      {xs.includes(0) && <line x1={sx(0)} x2={sx(0)} y1={P / 2} y2={H - P} stroke="#f97316" strokeDasharray="2 3" />}
      {series.map((s, k) => (
        <polyline key={k} fill="none" stroke={s.color} strokeWidth={1.6} strokeDasharray={s.dash ? "4 3" : undefined}
          points={s.values.map((v, i) => `${sx(xs[i])},${sy(v)}`).join(" ")} />
      ))}
      <text x={P} y={H - 8} fill="#8b95a7" fontSize="10">{x0}</text>
      <text x={W - P} y={H - 8} fill="#8b95a7" fontSize="10" textAnchor="end">{x1}</text>
      {xLabel && <text x={W / 2} y={H - 8} fill="#8b95a7" fontSize="10" textAnchor="middle">{xLabel}</text>}
      <text x={4} y={sy(y1) + 4} fill="#8b95a7" fontSize="10">{y1.toFixed(2)}</text>
      <text x={4} y={sy(y0)} fill="#8b95a7" fontSize="10">{y0.toFixed(2)}</text>
    </svg>
  );
}

export function BarsSvg({ values, labels, color = "#38bdf8", height = 120, refLine }: {
  values: number[]; labels?: string[]; color?: string; height?: number; refLine?: number;
}) {
  if (!values.length) return null;
  const W = 600, H = height, P = 18;
  const m = Math.max(...values.filter(isFinite), refLine ?? 0) || 1;
  const bw = (W - 2 * P) / values.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ direction: "ltr" }}>
      {values.map((v, i) => {
        const h = ((isFinite(v) ? v : 0) / m) * (H - 2 * P);
        return (
          <g key={i}>
            <rect x={P + i * bw + 1} y={H - P - h} width={Math.max(bw - 2, 1)} height={h} fill={color} opacity={0.85}>
              <title>{`${labels?.[i] ?? i}: ${v}`}</title>
            </rect>
            {labels && values.length <= 24 && <text x={P + i * bw + bw / 2} y={H - 4} fill="#8b95a7" fontSize="9" textAnchor="middle">{labels[i]}</text>}
          </g>
        );
      })}
      {refLine != null && <line x1={P} x2={W - P} y1={H - P - (refLine / m) * (H - 2 * P)} y2={H - P - (refLine / m) * (H - 2 * P)} stroke="#f97316" strokeDasharray="4 3" />}
    </svg>
  );
}

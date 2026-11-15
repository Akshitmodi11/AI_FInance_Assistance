// Recharts versions of the Plotly figures in src/web_app/components/charts.py — same data, same
// colors. Charts are pure: data in, chart out.

import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";
import type { Analysis, GoalPlan, PricePoint } from "../types";
import { pct, usd } from "../format";

export const PALETTE = ["#4c8df6", "#2ebd75", "#e9b949", "#f0524f", "#a77bf3", "#22b8cf", "#f0864f", "#8bc34a"];
export const SERIES = ["#2a78d6", "#eb6834"];
const BAND = { low: "#e9b949", expected: "#4c8df6", high: "#2ebd75" };

function Legend({ items }: { items: { name: string; color: string }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.name} style={{ ["--c" as string]: i.color }}>
          {i.name}
        </span>
      ))}
    </div>
  );
}

function tooltip(format: (v: number) => string, title?: (label: string | number | undefined) => string) {
  return ({ active, payload, label }: TooltipProps<number, string>) =>
    active && payload?.length ? (
      <div className="tooltip">
        {title && <div className="t-title">{title(label)}</div>}
        {payload
          .filter((p) => p.value != null)
          .map((p) => (
            <div key={String(p.name)}>
              <span style={{ color: p.color }}>●</span> {p.name}: <b>{format(p.value as number)}</b>
            </div>
          ))}
      </div>
    ) : null;
}

/** Donut of portfolio weights by holding, asset class or sector. */
export function Donut({ title, mix }: { title: string; mix: Record<string, number> }) {
  const data = Object.entries(mix).map(([name, value]) => ({ name, value }));
  return (
    <div>
      <div className="chart-title">{title}</div>
      <ResponsiveContainer width="100%" height={210}>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="92%" paddingAngle={1.5} stroke="none" isAnimationActive={false}>
            {data.map((_, i) => (
              <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
            ))}
          </Pie>
          <Tooltip content={tooltip((v) => pct(v, 1))} />
        </PieChart>
      </ResponsiveContainer>
      <Legend items={data.map((d, i) => ({ name: `${d.name} ${pct(d.value)}`, color: PALETTE[i % PALETTE.length] }))} />
    </div>
  );
}

/** Horizontal bars, largest first (sector exposure, chunk counts, …). */
export function HBars({
  title,
  values,
  format = (v) => String(v),
  color = SERIES[0],
}: {
  title: string;
  values: Record<string, number>;
  format?: (v: number) => string;
  color?: string;
}) {
  const data = Object.entries(values)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
  return (
    <div>
      <div className="chart-title">{title}</div>
      <ResponsiveContainer width="100%" height={Math.max(120, data.length * 30 + 20)}>
        <BarChart data={data} layout="vertical" margin={{ left: 0, right: 48, top: 4, bottom: 4 }}>
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="name" width={150} tickLine={false} axisLine={false} />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={tooltip(format)} />
          <Bar dataKey="value" name={title} fill={color} radius={[0, 4, 4, 0]} barSize={16} isAnimationActive={false}
            label={{ position: "right", formatter: format, fill: "var(--text-2)", fontSize: 12 }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** 0–100 semicircle gauge with red / amber / green bands. */
export function Gauge({ score, title }: { score: number; title: string }) {
  const r = 80;
  const arc = (from: number, to: number) => {
    const a = (v: number) => Math.PI * (1 - v / 100);
    const [x1, y1] = [100 + r * Math.cos(a(from)), 100 - r * Math.sin(a(from))];
    const [x2, y2] = [100 + r * Math.cos(a(to)), 100 - r * Math.sin(a(to))];
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };
  const tone = score < 40 ? "var(--bad)" : score < 70 ? "var(--warn)" : "var(--good)";
  return (
    <div>
      <div className="chart-title">{title}</div>
      <svg viewBox="0 0 200 120" width="100%" style={{ maxHeight: 200 }} role="img" aria-label={`${title}: ${score} out of 100`}>
        <path d={arc(0, 39.5)} stroke="var(--bad-soft)" strokeWidth={18} fill="none" />
        <path d={arc(40.5, 69.5)} stroke="var(--warn-soft)" strokeWidth={18} fill="none" />
        <path d={arc(70.5, 100)} stroke="var(--good-soft)" strokeWidth={18} fill="none" />
        {score > 0 && <path d={arc(0, Math.min(score, 100))} stroke={tone} strokeWidth={7} strokeLinecap="round" fill="none" />}
        <text x={100} y={96} textAnchor="middle" fontSize={30} fontWeight={650} fill="var(--text)">
          {score}
        </text>
        <text x={100} y={114} textAnchor="middle" fontSize={11} fill="var(--muted)">
          out of 100
        </text>
      </svg>
    </div>
  );
}

/** Close prices (green if the shown period is up, red if down) with 20- and 50-day moving
 *  averages computed on the full history, so a 1-month view still shows a real 50-day average. */
export function PriceChart({
  history,
  ticker,
  showLast,
  bare,
}: {
  history: PricePoint[];
  ticker: string;
  showLast?: number;
  bare?: boolean;
}) {
  const closes = history.map((h) => h.close);
  const sma = (n: number, j: number) =>
    j >= n - 1 ? closes.slice(j - n + 1, j + 1).reduce((a, b) => a + b, 0) / n : null;
  const start = showLast ? Math.max(history.length - showLast, 0) : 0;
  const data = history.slice(start).map((h, i) => ({
    date: h.date,
    close: h.close,
    sma20: sma(20, start + i),
    sma50: sma(50, start + i),
  }));
  const up = data.length < 2 || data[data.length - 1].close >= data[0].close;
  const price = up ? "var(--good)" : "var(--bad)";
  const gid = `px-${ticker.replace(/\W/g, "")}-${up ? "u" : "d"}`;
  const lines = [
    { key: "sma20", name: "SMA 20", color: "#e9b949" },
    { key: "sma50", name: "SMA 50", color: "#a77bf3" },
  ];
  const fmtDate = (d: string | number | undefined) =>
    new Date(String(d)).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "2-digit" });
  return (
    <div>
      {!bare && <div className="chart-title">{ticker} — price and moving averages</div>}
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ left: 0, right: 8, top: 6, bottom: 0 }}>
          <defs>
            <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={price} stopOpacity={0.22} />
              <stop offset="100%" stopColor={price} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="date" tickFormatter={(d) => fmtDate(d).replace(/, \d+$/, "")} minTickGap={48} tickLine={false} axisLine={false} />
          <YAxis domain={["auto", "auto"]} orientation="right" tickFormatter={(v) => usd(v)} width={60} tickLine={false} axisLine={false} />
          <Tooltip content={tooltip((v) => usd(v, 2), fmtDate)} cursor={{ stroke: "var(--border-strong)", strokeDasharray: "3 3" }} />
          <Area dataKey="close" name={`${ticker} close`} stroke={price} strokeWidth={2} fill={`url(#${gid})`} dot={false} isAnimationActive={false} activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--surface)" }} />
          {lines.map((l) => (
            <Line key={l.key} dataKey={l.key} name={l.name} stroke={l.color} strokeWidth={1.4} strokeDasharray="4 3" dot={false} isAnimationActive={false} connectNulls />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
      <Legend items={[{ name: `${ticker} close`, color: price }, ...lines]} />
    </div>
  );
}

/** Savings path in low / expected / high scenarios vs the inflation-adjusted target. */
export function GoalChart({ plan }: { plan: GoalPlan }) {
  const data = plan.yearly_projection.map((r) => ({ ...r, band: [r.low, r.high] }));
  const target = plan.target_in_future_dollars;
  return (
    <div>
      <div className="chart-title">Savings projection by return scenario</div>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ left: 0, right: 8, top: 16, bottom: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="year" tickFormatter={(y) => `yr ${y}`} tickLine={false} axisLine={false} />
          <YAxis tickFormatter={(v) => (v >= 1000 ? `$${Math.round(v / 1000)}k` : usd(v))} width={56} tickLine={false} axisLine={false} />
          <Tooltip content={tooltip((v) => usd(v), (y) => `Year ${y}`)} />
          <Area dataKey="band" name="Low – high range" fill="var(--accent-soft)" stroke="none" isAnimationActive={false} legendType="none" tooltipType="none" />
          <Line dataKey="high" name="High return" stroke={BAND.high} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          <Line dataKey="low" name="Low return" stroke={BAND.low} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          <Line dataKey="expected" name="Expected" stroke={BAND.expected} strokeWidth={3} dot={false} isAnimationActive={false} />
          <ReferenceLine y={target} stroke="var(--bad)" strokeDasharray="5 4"
            label={{ value: `Target ${usd(target)}`, position: "insideTopLeft", fill: "var(--bad)", fontSize: 12 }} />
        </ComposedChart>
      </ResponsiveContainer>
      <Legend items={[
        { name: "Expected", color: BAND.expected },
        { name: "High return", color: BAND.high },
        { name: "Low return", color: BAND.low },
        { name: "Target (inflation-adjusted)", color: "var(--bad)" },
      ]} />
    </div>
  );
}

/** Grouped bars of two 0–1 measures per category (Recall@5 and MRR per retrieval mode). */
export function GroupedBars({
  title,
  rows,
  series,
}: {
  title: string;
  rows: Record<string, string | number>[];
  series: { key: string; name: string }[];
}) {
  return (
    <div>
      <div className="chart-title">{title}</div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={rows} margin={{ left: 0, right: 8, top: 18, bottom: 0 }} barGap={2}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="mode" tickLine={false} axisLine={false} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} width={34} tickLine={false} axisLine={false} />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={tooltip((v) => v.toFixed(2))} />
          {series.map((s, i) => (
            <Bar key={s.key} dataKey={s.key} name={s.name} fill={SERIES[i]} radius={[4, 4, 0, 0]} maxBarSize={34} isAnimationActive={false}
              label={{ position: "top", formatter: (v: number) => v.toFixed(2), fill: "var(--text-2)", fontSize: 11 }} />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <Legend items={series.map((s, i) => ({ name: s.name, color: SERIES[i] }))} />
    </div>
  );
}

export function AnalysisCharts({ a }: { a: Analysis }) {
  const byHolding = Object.fromEntries(a.positions.map((p) => [p.ticker, p.weight]));
  return (
    <div className="grid grid-3">
      <div className="card"><Donut title="Allocation by holding" mix={byHolding} /></div>
      <div className="card"><Donut title="Allocation by asset class" mix={a.asset_class_mix} /></div>
      <div className="card"><Gauge title="Diversification score (educational)" score={a.diversification_score} /></div>
    </div>
  );
}

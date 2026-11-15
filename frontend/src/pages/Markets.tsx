import { useEffect, useState } from "react";
import { api } from "../api";
import { PriceChart } from "../components/charts";
import { Icon } from "../components/icons";
import { ErrorNote, Loading, Metric, PageHead, Segmented, SourceBadge } from "../components/ui";
import { pct, signedPct, usd } from "../format";
import type { IndexQuote, Lookup } from "../types";

const TREND = {
  uptrend: { label: "Uptrend", tone: "good", text: "Price is above its 20-day average, which is above the 50-day." },
  downtrend: { label: "Downtrend", tone: "bad", text: "Price is below its 20-day average, which is below the 50-day." },
  sideways: { label: "Sideways", tone: "", text: "The moving averages don't line up in one direction." },
  unknown: { label: "Not enough history", tone: "warn", text: "There isn't enough price history to judge the trend." },
} as const;
const RANGES = ["1M", "3M", "6M", "1Y"] as const;
const DAYS = { "1M": 21, "3M": 63, "6M": 126, "1Y": 252 };
const POPULAR = ["NVDA", "Apple", "Tesla", "VOO", "Microsoft"];

export function Markets({ indexes }: { indexes: IndexQuote[] }) {
  const [query, setQuery] = useState("VTI");
  const [data, setData] = useState<Lookup | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [range, setRange] = useState<(typeof RANGES)[number]>("6M");

  async function lookup(q: string) {
    if (!q.trim()) return;
    setQuery(q);
    setBusy(true);
    setError(null);
    try {
      setData(await api.lookup(q));
    } catch (e) {
      setError((e as Error).message);
      setData(null);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    lookup("VTI");
  }, []);

  const q = data?.snapshot.quote;
  const trend = data ? TREND[data.snapshot.trend] : null;
  return (
    <>
      <PageHead title="Markets" sub="Live quotes with source and freshness labels. Search by ticker or company name." />
      <div className="kpis">
        {indexes.length === 0 ? <Loading /> : indexes.map((i) => (
          <Metric key={i.ticker} label={`${i.name} · ${i.ticker}`}
            value={i.price == null ? "n/a" : usd(i.price, 2)}
            delta={i.change_pct == null ? undefined : signedPct(i.change_pct)}
            tone={(i.change_pct ?? 0) >= 0 ? "up" : "down"} />
        ))}
      </div>

      <div className="card" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <form className="row" onSubmit={(e) => { e.preventDefault(); lookup(query); }}>
          <label className="field grow">
            Ticker or company
            <input className="input" value={query} maxLength={40} placeholder="NVDA or nvidia" onChange={(e) => setQuery(e.target.value)} />
          </label>
          <button className="btn primary" disabled={busy}><Icon name="search" /> Look up</button>
        </form>
        <div className="meta-row">
          <span className="caption">Popular:</span>
          {POPULAR.map((p) => (
            <button key={p} className="btn small" type="button" onClick={() => lookup(p)}>{p}</button>
          ))}
          {busy && <Loading text="Loading…" />}
        </div>
      </div>
      <ErrorNote error={error} />

      {data && q && trend && (
        <>
          <div className="card" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="chart-head" style={{ alignItems: "flex-end" }}>
              <div>
                <div className="caption">{data.name ?? data.ticker}</div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 20, fontWeight: 650 }}>{data.ticker}</span>
                  <span className="num" style={{ fontSize: 26, fontWeight: 600 }}>{usd(q.price, 2)}</span>
                  <span className={`num ${q.change >= 0 ? "up" : "down"}`} style={{ fontWeight: 500 }}>
                    {q.change >= 0 ? "▲" : "▼"} {q.change >= 0 ? "+" : ""}{q.change.toFixed(2)} ({signedPct(q.change_pct)})
                  </span>
                  <span className={`badge ${trend.tone === "good" ? "accent" : ""}`}
                    style={trend.tone === "good" ? { background: "var(--good-soft)", color: "var(--good)" } : trend.tone === "bad" ? { background: "var(--bad-soft)", color: "var(--bad)" } : undefined}>
                    {trend.label}
                  </span>
                </div>
              </div>
              <Segmented inline label="Range" options={RANGES} value={range} onChange={setRange} />
            </div>
            <PriceChart history={data.history} ticker={data.ticker} showLast={DAYS[range]} bare />
            <SourceBadge source={q.source} asOf={q.as_of} cached={q.is_cached} mock={q.is_mock} />
          </div>
          <div className="kpis">
            <Metric label="52-week low" value={data.snapshot.low_52w == null ? "—" : usd(data.snapshot.low_52w, 2)} />
            <Metric label="52-week high" value={data.snapshot.high_52w == null ? "—" : usd(data.snapshot.high_52w, 2)} />
            <Metric label="1-month return" value={pct(data.snapshot.return_1m, 1)}
              tone={(data.snapshot.return_1m ?? 0) >= 0 ? "up" : "down"} />
            <Metric label="6-month return" value={pct(data.snapshot.return_6m, 1)}
              tone={(data.snapshot.return_6m ?? 0) >= 0 ? "up" : "down"} />
          </div>
          <div className="note">
            <Icon name="info" />
            <span>
              <b>{trend.label}.</b> {trend.text} Moving averages smooth daily noise: the 20-day shows the recent direction,
              the 50-day the medium-term one. They describe the past — they don't predict prices.
            </span>
          </div>
        </>
      )}
    </>
  );
}

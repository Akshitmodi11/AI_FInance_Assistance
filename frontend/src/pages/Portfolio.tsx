import { useEffect, useState } from "react";
import { api } from "../api";
import { AnalysisCharts, HBars } from "../components/charts";
import { ErrorNote, Loading, Metric, PageHead, SourceBadge } from "../components/ui";
import { pct, usd } from "../format";
import type { Analysis, Holding } from "../types";

type Row = { ticker: string; shares: string };
const toRows = (h: Holding[]): Row[] => h.map((x) => ({ ticker: x.ticker, shares: String(x.shares) }));

function parseCsv(text: string): Holding[] {
  const [head, ...lines] = text.trim().split(/\r?\n/);
  const cols = head.split(",").map((c) => c.trim().toLowerCase());
  const ti = cols.indexOf("ticker");
  const si = cols.indexOf("shares");
  if (ti < 0 || si < 0) throw new Error("The CSV needs `ticker` and `shares` columns.");
  return lines
    .map((l) => l.split(","))
    .map((c) => ({ ticker: c[ti]?.trim() ?? "", shares: Number(c[si]) }))
    .filter((h) => h.ticker && h.shares > 0);
}

export function Portfolio({
  analysis,
  onAnalyzed,
}: {
  analysis: Analysis | null;
  onAnalyzed: (holdings: Holding[], a: Analysis) => void;
}) {
  const [rows, setRows] = useState<Row[]>(
    toRows(analysis?.positions ?? [
      { ticker: "VTI", shares: 20 },
      { ticker: "VXUS", shares: 30 },
      { ticker: "BND", shares: 25 },
      { ticker: "AAPL", shares: 5 },
    ]),
  );
  const [samples, setSamples] = useState<Record<string, Holding[]>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.samples().then(setSamples).catch(() => {});
  }, []);

  const update = (i: number, patch: Partial<Row>) => setRows((r) => r.map((x, j) => (j === i ? { ...x, ...patch } : x)));

  async function analyze() {
    const holdings = rows
      .map((r) => ({ ticker: r.ticker.trim().toUpperCase(), shares: Number(r.shares) }))
      .filter((h) => h.ticker && h.shares > 0);
    if (!holdings.length) return setError("Add at least one holding with shares > 0.");
    setBusy(true);
    setError(null);
    try {
      onAnalyzed(holdings, await api.analyze(holdings));
    } catch (e) {
      setError(`Couldn't analyze the portfolio: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHead title="Portfolio analysis" sub="Educational analysis of diversification, concentration, costs and risk — not advice." />
      <div className="grid grid-2" style={{ gridTemplateColumns: "minmax(0,3fr) minmax(0,2fr)" }}>
        <div className="card">
          <h3>Holdings</h3>
          <table className="table">
            <thead>
              <tr><th>Ticker</th><th>Shares</th><th /></tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td><input className="input" value={r.ticker} aria-label="Ticker" onChange={(e) => update(i, { ticker: e.target.value })} /></td>
                  <td><input className="input" type="number" min={0} step="any" value={r.shares} aria-label="Shares" onChange={(e) => update(i, { shares: e.target.value })} /></td>
                  <td className="num">
                    <button className="btn small" aria-label={`Remove ${r.ticker}`} onClick={() => setRows((x) => x.filter((_, j) => j !== i))}>✕</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="row" style={{ marginTop: 10 }}>
            <button className="btn" onClick={() => setRows((r) => [...r, { ticker: "", shares: "" }])}>+ Add holding</button>
            <button className="btn primary" onClick={analyze} disabled={busy}>Analyze portfolio</button>
            {busy && <Loading text="Fetching prices…" />}
          </div>
        </div>
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <label className="field">
            Start from a sample
            <select className="input" defaultValue="" onChange={(e) => samples[e.target.value] && setRows(toRows(samples[e.target.value]))}>
              <option value="">—</option>
              {Object.keys(samples).map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label className="field">
            …or upload a CSV (columns: ticker, shares)
            <input className="input" type="file" accept=".csv" onChange={async (e) => {
              const f = e.target.files?.[0];
              if (!f) return;
              try { setRows(toRows(parseCsv(await f.text()))); setError(null); } catch (err) { setError((err as Error).message); }
            }} />
          </label>
          <div className="caption">After analyzing, the Chat tab can answer questions like “Is my portfolio diversified enough?”</div>
        </div>
      </div>
      <ErrorNote error={error} />
      {analysis && <Results a={analysis} />}
    </>
  );
}

function Results({ a }: { a: Analysis }) {
  return (
    <>
      <div className="kpis">
        <Metric label="Total value" value={usd(a.total_value)} />
        <Metric label="Holdings behave like" value={a.effective_n.toFixed(1)} hint="1 / HHI of weights: how many equal-sized holdings this acts like" />
        <Metric label="Largest holding" value={pct(a.top_holding_weight)} />
        <Metric label="Fund costs / yr" value={pct(a.weighted_expense_ratio, 2)} />
        <Metric label="Volatility (1y)" value={pct(a.annualized_volatility)} />
      </div>
      <SourceBadge source={a.data_source} mock={a.is_mock} />
      <AnalysisCharts a={a} />
      <div className="grid grid-2">
        <div className="card"><HBars title="Sector exposure" values={a.sector_mix} format={(v) => pct(v)} /></div>
        <div className="card">
          <h3>Observations <span className="caption">(educational)</span></h3>
          <ul style={{ margin: 0, paddingLeft: 20 }}>{a.observations.map((o) => <li key={o}>{o}</li>)}</ul>
        </div>
      </div>
      <details className="more">
        <summary>Positions</summary>
        <div className="scroll-x">
          <table className="table">
            <thead><tr><th>Ticker</th><th className="num">Shares</th><th className="num">Price</th><th className="num">Value</th><th className="num">Weight</th><th>Asset class</th><th>Sector</th><th className="num">Expense ratio</th></tr></thead>
            <tbody>
              {a.positions.map((p) => (
                <tr key={p.ticker}>
                  <td><b>{p.ticker}</b></td><td className="num">{p.shares}</td><td className="num">{usd(p.price, 2)}</td>
                  <td className="num">{usd(p.value)}</td><td className="num">{pct(p.weight, 1)}</td><td>{p.asset_class}</td><td>{p.sector}</td>
                  <td className="num">{pct(p.expense_ratio, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

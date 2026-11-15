import { useEffect, useState } from "react";
import { api } from "../api";
import { GroupedBars, HBars } from "../components/charts";
import { ErrorNote, Loading, Metric, PageHead } from "../components/ui";
import { pct } from "../format";
import type { Metrics, Status } from "../types";

const MODE_LABELS: Record<string, string> = {
  dense: "dense",
  sparse: "sparse",
  hybrid: "hybrid (RRF)",
  rerank: "hybrid → rerank",
};
const label = (m: string) => MODE_LABELS[m] ?? m;

export function Insights({ status }: { status: Status | null }) {
  const [m, setM] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.metrics().then(setM).catch((e) => setError(e.message));
  }, []);

  if (error) return <><PageHead title="Insights" /><ErrorNote error={error} /></>;
  if (!m) return <><PageHead title="Insights" /><Loading /></>;

  const rows = m.retrieval.rows;
  const active = rows.find((r) => r.mode === m.config.retrieval_mode) ?? rows[0];
  const router = m.routing.rows[0];
  const guard = m.safety.rows[0]; // the 18-prompt set; the other rows are subsets of it
  const costs = m.latency.rows.map((r) => r.cost_usd);
  const types = [...new Set(rows.flatMap((r) => Object.keys(r.by_type)))].sort();

  return (
    <>
      <PageHead
        title="Insights"
        sub={<>Every measured number in one place. Snapshot {m.generated}: retrieval and knowledge base measured then; routing, safety, latency and tests from their last recorded run (dates below).</>}
      />
      <div className="kpis">
        <Metric label={`Recall@${m.retrieval.k}`} value={active.recall.toFixed(2)} hint={`${label(active.mode)}, ${m.retrieval.questions} labelled questions`} />
        <Metric label="MRR" value={active.mrr.toFixed(2)} hint="How high the first correct source ranks" />
        <Metric label="Routing accuracy" value={pct(router.exact)} hint={router.router} />
        <Metric label="Safety prompts" value={`${guard.passed}/${guard.total}`} hint={guard.check} />
        <Metric label="Tests · coverage" value={`${m.tests.passed} · ${pct(m.tests.coverage)}`} />
        <Metric label="Cost / question" value={`$${Math.min(...costs).toFixed(4)}–${Math.max(...costs).toFixed(4)}`} />
      </div>

      <h3>Retrieval — which search mode finds the right source</h3>
      <div className="grid grid-2" style={{ gridTemplateColumns: "minmax(0,3fr) minmax(0,2fr)" }}>
        <div className="card">
          <GroupedBars title={`Recall@${m.retrieval.k} and MRR by mode (${m.retrieval.questions} questions)`}
            rows={rows.map((r) => ({ mode: label(r.mode), recall: r.recall, mrr: r.mrr }))}
            series={[{ key: "recall", name: `Recall@${m.retrieval.k}` }, { key: "mrr", name: "MRR" }]} />
        </div>
        <div className="card scroll-x">
          <table className="table">
            <thead><tr><th>Mode</th>{types.map((t) => <th key={t} className="num">R@5 {t}</th>)}<th className="num">Off-topic refused</th><th className="num">p50 ms</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.mode} style={r === active ? { fontWeight: 600 } : undefined}>
                  <td>{label(r.mode)}</td>
                  {types.map((t) => <td key={t} className="num">{(r.by_type[t] ?? 0).toFixed(2)}</td>)}
                  <td className="num">{pct(r.no_answer)}</td><td className="num">{Math.round(r.p50_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="caption" style={{ marginTop: 8 }}>Active mode: <b>{label(active.mode)}</b></div>
        </div>
      </div>
      <details className="more">
        <summary>Missed questions per mode</summary>
        {rows.map((r) => (
          <div key={r.mode} style={{ marginBottom: 8 }}>
            <b>{label(r.mode)}</b> — {r.misses.length} misses
            <pre style={{ margin: "4px 0 0", whiteSpace: "pre-wrap", fontSize: 12.5, color: "var(--text-2)" }}>{r.misses.join("\n") || "none"}</pre>
          </div>
        ))}
      </details>

      <h3>Answers — routing, safety, speed and cost</h3>
      <div className="grid grid-3" style={{ gridTemplateColumns: "minmax(0,2fr) minmax(0,2fr) minmax(0,3fr)" }}>
        <div className="card">
          <div className="caption">Routing · {m.routing.questions} questions · {m.routing.date}</div>
          <table className="table">
            <thead><tr><th>Router</th><th className="num">Exact</th><th className="num">Primary</th></tr></thead>
            <tbody>{m.routing.rows.map((r) => <tr key={r.router}><td>{r.router}</td><td className="num">{pct(r.exact)}</td><td className="num">{pct(r.primary)}</td></tr>)}</tbody>
          </table>
        </div>
        <div className="card">
          <div className="caption">Safety · {m.safety.date}</div>
          <table className="table">
            <thead><tr><th>Check</th><th className="num">Passed</th></tr></thead>
            <tbody>{m.safety.rows.map((r) => <tr key={r.check}><td>{r.check}</td><td className="num">{r.passed}/{r.total}</td></tr>)}</tbody>
          </table>
        </div>
        <div className="card scroll-x">
          <div className="caption">End-to-end latency and cost per question · {m.latency.date}</div>
          <table className="table">
            <thead><tr><th>Agent</th><th className="num">p50 s</th><th className="num">p95 s</th><th className="num">Tokens</th><th className="num">Cost $</th></tr></thead>
            <tbody>{m.latency.rows.map((r) => <tr key={r.intent}><td>{r.intent}</td><td className="num">{r.p50_s}</td><td className="num">{r.p95_s}</td><td className="num">{r.tokens.toLocaleString()}</td><td className="num">{r.cost_usd.toFixed(4)}</td></tr>)}</tbody>
          </table>
        </div>
      </div>

      <h3>Knowledge base — {m.kb.chunks.toLocaleString()} chunks from {m.kb.sources} sources</h3>
      <div className="grid grid-2">
        <div className="card"><HBars title="Chunks by category" values={m.kb.by_category} /></div>
        <div className="card"><HBars title="Chunks by file type" values={m.kb.by_file_type} /></div>
      </div>

      <h3>Experiment log</h3>
      <div className="card scroll-x">
        <table className="table">
          <thead><tr><th>Date</th><th>Experiment</th><th>Result</th><th>Decision</th></tr></thead>
          <tbody>{m.experiments.map((e) => <tr key={e.experiment}><td style={{ whiteSpace: "nowrap" }}>{e.date}</td><td>{e.experiment}</td><td>{e.result}</td><td>{e.decision}</td></tr>)}</tbody>
        </table>
      </div>

      <h3>System</h3>
      <div className="card">
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          <li>Retrieval: <b>{label(m.config.retrieval_mode)}</b> on <b>{m.config.vector_store}</b>, gate ≥ {m.config.min_score}</li>
          <li>Embeddings: <code>{m.config.embedding_model}</code> · Reranker: <code>{m.config.rerank_model}</code></li>
          <li>LLM provider: <b>{m.config.llm_provider}</b></li>
          <li>Traces: {status?.services["Tracing (LangSmith)"] ? <a href="https://smith.langchain.com" target="_blank" rel="noreferrer">LangSmith → project “finnie”</a> : "off (no key)"}</li>
          <li>API: <a href="/docs" target="_blank" rel="noreferrer">interactive docs at /docs</a></li>
        </ul>
      </div>
    </>
  );
}

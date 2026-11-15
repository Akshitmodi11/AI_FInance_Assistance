import { useEffect, useMemo, useState, type FormEvent } from "react";
import { api } from "../api";
import { ErrorNote, Loading, Metric, PageHead } from "../components/ui";
import type { SearchResult, Source, Status } from "../types";

export function Knowledge({ status }: { status: Status | null }) {
  const [kb, setKb] = useState<{ chunks: number; sources: Source[] } | null>(null);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.sources().then(setKb).catch((e) => setError(e.message));
  }, []);

  const grouped = useMemo(() => {
    const g: Record<string, Source[]> = {};
    kb?.sources.forEach((s) => (g[s.category] ??= []).push(s));
    return g;
  }, [kb]);

  async function search(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.search(query, category));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const mode = status?.retrieval_mode;
  return (
    <>
      <PageHead title="Knowledge base" sub="The cited sources behind Finnie's answers — IRS publications, SEC/FINRA guides and plain-language articles." />
      <div className="kpis">
        <Metric label="Sources" value={kb?.sources.length ?? "…"} />
        <Metric label="Indexed chunks" value={kb?.chunks.toLocaleString() ?? "…"} />
        <Metric label="Retrieval" value={mode === "rerank" ? "Hybrid + rerank" : mode ? mode[0].toUpperCase() + mode.slice(1) : "…"} />
      </div>

      <form className="row" onSubmit={search}>
        <label className="field grow">
          Search the knowledge base
          <input className="input" value={query} placeholder="e.g. HSA contribution limit" onChange={(e) => setQuery(e.target.value)} />
        </label>
        <label className="field">
          Category
          <select className="input" value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">All</option>
            {status?.categories.map((c) => <option key={c}>{c}</option>)}
          </select>
        </label>
        <button className="btn primary" disabled={busy}>Search</button>
        {busy && <Loading text="Searching…" />}
      </form>
      <ErrorNote error={error} />

      {result && !result.confident && (
        <div className="note warn">
          No confident match (best similarity {(result.top_similarity ?? 0).toFixed(2)} &lt; {result.min_score}). Finnie would say it doesn't know.
        </div>
      )}
      {result?.results.map((r, i) => (
        <div key={i} className="card">
          <div><a href={r.url} target="_blank" rel="noreferrer"><b>{r.citation}</b></a> <span className="caption">· {r.category}</span></div>
          <div className="caption">relevance {r.score.toFixed(3)} · similarity {r.similarity.toFixed(2)}</div>
          <p style={{ margin: "6px 0 0", color: "var(--text-2)", whiteSpace: "pre-line" }}>
            {r.text.length > 700 ? `${r.text.slice(0, 700)}…` : r.text}
          </p>
        </div>
      ))}

      <h3>Browse sources</h3>
      {!kb && !error && <Loading />}
      {Object.entries(grouped).map(([cat, list]) => (
        <details key={cat} className="more">
          <summary>{cat} ({list.length})</summary>
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            {list.map((s) => (
              <li key={s.url + s.title}>
                <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>{" "}
                <span className="caption">— {s.source}, {s.file_type.toUpperCase()}, {s.chunks} chunks{s.tax_year ? `, tax year ${s.tax_year}` : ""}</span>
              </li>
            ))}
          </ul>
        </details>
      ))}
    </>
  );
}

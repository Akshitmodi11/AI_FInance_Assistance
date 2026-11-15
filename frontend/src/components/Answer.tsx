import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatAnswer } from "../types";
import { AGENTS } from "../format";
import { AnalysisCharts, GoalChart, PriceChart } from "./charts";
import { Icon } from "./icons";
import { SourceBadge } from "./ui";

export function Markdown({ text }: { text: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{ a: ({ node: _node, ...p }) => <a {...p} target="_blank" rel="noreferrer" /> }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

/** One assistant turn: agents used, guardrail note, answer, charts, sources, disclaimer. */
export function Answer({ a }: { a: ChatAnswer }) {
  const { portfolio_analysis, goal_plan, market } = a.data ?? {};
  return (
    <div className="answer">
      <div className="meta-row">
        {a.agents.length ? (
          a.agents.map((ag) => (
            <span key={ag} className="badge" style={{ ["--c" as string]: AGENTS[ag]?.color }}>
              <span className="pip" /> {AGENTS[ag]?.label ?? ag}
            </span>
          ))
        ) : (
          <span className="badge"><Icon name="shield" size={12} /> Guardrail</span>
        )}
      </div>
      {a.guardrail_note && (
        <div className="note">
          <Icon name="shield" /> <span>{a.guardrail_note}</span>
        </div>
      )}
      <Markdown text={a.answer} />
      {portfolio_analysis && <AnalysisCharts a={portfolio_analysis} />}
      {goal_plan && (
        <div className="card">
          <GoalChart plan={goal_plan} />
        </div>
      )}
      {market?.map((m) => (
        <div key={m.snapshot.quote.ticker} className="card">
          <PriceChart history={m.history} ticker={m.snapshot.quote.ticker} showLast={126} />
          <SourceBadge source={m.snapshot.quote.source} asOf={m.snapshot.quote.as_of} cached={m.snapshot.quote.is_cached} mock={m.snapshot.quote.is_mock} />
        </div>
      ))}
      {a.citations.length > 0 && (
        <details className="more">
          <summary>Sources ({a.citations.length})</summary>
          <ol style={{ margin: 0, paddingLeft: 20, display: "flex", flexDirection: "column", gap: 4 }}>
            {a.citations.map((c, i) => (
              <li key={i}>
                <a href={c.url} target="_blank" rel="noreferrer">{c.label}</a>{" "}
                <span className="caption">· {c.source}</span>
              </li>
            ))}
          </ol>
        </details>
      )}
      <div className="disclaimer">{a.disclaimer}</div>
    </div>
  );
}

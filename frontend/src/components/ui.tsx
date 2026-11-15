import type { ReactNode } from "react";
import { Icon } from "./icons";

export function Metric({
  label,
  value,
  delta,
  tone,
  hint,
}: {
  label: string;
  value: ReactNode;
  delta?: ReactNode;
  tone?: "up" | "down";
  hint?: string;
}) {
  return (
    <div className="card metric" title={hint}>
      <span className="label">{label}</span>
      <span className={`value ${delta == null && tone ? tone : ""}`}>{value}</span>
      {delta != null && (
        <span className={`delta ${tone ?? ""}`}>
          {tone === "up" ? "▲ " : tone === "down" ? "▼ " : ""}
          {delta}
        </span>
      )}
    </div>
  );
}

/** One-line freshness label shown next to every number from a market-data provider. */
export function SourceBadge({
  source,
  asOf,
  cached,
  mock,
}: {
  source: string;
  asOf?: string;
  cached?: boolean;
  mock?: boolean;
}) {
  return (
    <div className="meta-row caption">
      <span className="badge">Source · {source.replace(/_/g, " ")}</span>
      {asOf && <span>as of {asOf.slice(0, 16).replace("T", " ")} UTC</span>}
      {cached && <span className="badge">Cached ≤ 30 min</span>}
      {mock && <span className="badge mock">Sample data — live providers unavailable</span>}
    </div>
  );
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
  inline,
}: {
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  inline?: boolean;
}) {
  return (
    <div className={`segmented ${inline ? "inline" : ""}`} role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o}
          type="button"
          role="radio"
          aria-checked={o === value}
          className={o === value ? "on" : ""}
          onClick={() => onChange(o)}
        >
          {o[0].toUpperCase() + o.slice(1)}
        </button>
      ))}
    </div>
  );
}

export function Loading({ text = "Loading…" }: { text?: string }) {
  return (
    <div className="loading">
      <span className="spinner" /> {text}
    </div>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  return error ? (
    <div className="note bad">
      <Icon name="alert" /> <span>{error}</span>
    </div>
  ) : null;
}

export function PageHead({ title, sub, right }: { title: string; sub?: ReactNode; right?: ReactNode }) {
  return (
    <div className="page-head" style={right ? { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-end", gap: 12 } : undefined}>
      <div>
        <h2>{title}</h2>
        {sub && <p>{sub}</p>}
      </div>
      {right}
    </div>
  );
}

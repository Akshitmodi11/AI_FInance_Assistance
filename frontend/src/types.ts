// Shapes returned by the FastAPI backend (src/api/main.py → src/core/schemas.py).

export type Level = "beginner" | "intermediate" | "advanced";
export type Risk = "conservative" | "moderate" | "aggressive";

export interface Profile {
  knowledge_level: Level;
  risk_tolerance: Risk;
  horizon_years: number;
}

export interface Holding {
  ticker: string;
  shares: number;
}

export interface Status {
  services: Record<string, boolean>;
  risk_profiles: Risk[];
  categories: string[];
  retrieval_mode: string;
  disclaimer: string;
}

export interface Quote {
  ticker: string;
  price: number;
  change: number;
  change_pct: number;
  as_of: string;
  source: string;
  is_cached: boolean;
  is_mock: boolean;
}

export interface Snapshot {
  quote: Quote;
  high_52w: number | null;
  low_52w: number | null;
  sma_20: number | null;
  sma_50: number | null;
  trend: "uptrend" | "downtrend" | "sideways" | "unknown";
  return_1m: number | null;
  return_6m: number | null;
}

export interface PricePoint {
  date: string;
  close: number;
}

export interface Lookup {
  ticker: string;
  name: string | null;
  snapshot: Snapshot;
  history: PricePoint[];
}

export interface IndexQuote extends Partial<Omit<Quote, "price">> {
  name: string;
  ticker: string;
  price: number | null;
}

export interface Position {
  ticker: string;
  shares: number;
  price: number;
  value: number;
  weight: number;
  asset_class: string;
  sector: string;
  expense_ratio: number | null;
}

export interface Analysis {
  total_value: number;
  positions: Position[];
  effective_n: number;
  top_holding_weight: number;
  asset_class_mix: Record<string, number>;
  sector_mix: Record<string, number>;
  weighted_expense_ratio: number | null;
  annualized_volatility: number | null;
  diversification_score: number;
  observations: string[];
  data_source: string;
  is_mock: boolean;
}

export interface GoalInput {
  goal_name: string;
  target_amount: number;
  years: number;
  current_savings: number;
  monthly_contribution: number | null;
  risk_profile: Risk;
}

export interface GoalPlan {
  goal: GoalInput;
  inflation: number;
  target_in_future_dollars: number;
  scenarios: {
    label: "low" | "expected" | "high";
    annual_return: number;
    required_monthly: number;
    projected_value: number | null;
    on_track: boolean | null;
  }[];
  yearly_projection: { year: number; low: number; expected: number; high: number }[];
}

export interface Citation {
  label: string;
  url: string;
  source: string;
}

export interface ChatAnswer {
  thread_id: string;
  answer: string;
  agents: string[];
  citations: Citation[];
  data: {
    portfolio_analysis?: Analysis;
    goal_plan?: GoalPlan;
    market?: { snapshot: Snapshot; history: PricePoint[] }[];
  };
  guardrail_note: string | null;
  disclaimer: string;
}

export type ChatTurn = { role: "user"; content: string } | { role: "assistant"; answer: ChatAnswer };

export interface SearchResult {
  confident: boolean;
  top_similarity: number | null;
  min_score: number;
  results: {
    citation: string;
    url: string;
    category: string;
    score: number;
    similarity: number;
    text: string;
  }[];
}

export interface Source {
  title: string;
  url: string;
  source: string;
  category: string;
  file_type: string;
  tax_year: number | null;
  chunks: number;
}

export interface RetrievalRow {
  mode: string;
  recall: number;
  mrr: number;
  no_answer: number;
  by_type: Record<string, number>;
  p50_ms: number;
  misses: string[];
}

export interface Metrics {
  generated: string;
  config: {
    retrieval_mode: string;
    vector_store: string;
    embedding_model: string;
    rerank_model: string;
    min_score: number;
    llm_provider: string;
  };
  kb: {
    chunks: number;
    sources: number;
    by_category: Record<string, number>;
    by_file_type: Record<string, number>;
  };
  retrieval: { questions: number; k: number; rows: RetrievalRow[] };
  routing: { date: string; questions: number; rows: { router: string; exact: number; primary: number }[] };
  safety: { date: string; rows: { check: string; passed: number; total: number }[] };
  latency: {
    date: string;
    rows: { intent: string; p50_s: number; p95_s: number; tokens: number; cost_usd: number }[];
  };
  tests: { date: string; passed: number; coverage: number; gate: number };
  experiments: { date: string; experiment: string; result: string; decision: string }[];
}

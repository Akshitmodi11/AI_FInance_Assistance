// Thin fetch wrapper around the FastAPI backend. Errors carry the server's `detail` message.

import type {
  Analysis,
  ChatAnswer,
  GoalInput,
  GoalPlan,
  Holding,
  IndexQuote,
  Lookup,
  Metrics,
  Profile,
  SearchResult,
  Source,
  Status,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  status: () => request<Status>("/status"),
  chat: (question: string, thread_id: string, profile: Profile, portfolio: Holding[] | null) =>
    post<ChatAnswer>("/chat", {
      question,
      thread_id,
      profile,
      portfolio: portfolio ? { holdings: portfolio } : null,
    }),
  indexes: () => request<IndexQuote[]>("/market/indexes"),
  lookup: (q: string) => request<Lookup>(`/market/lookup?q=${encodeURIComponent(q)}`),
  samples: () => request<Record<string, Holding[]>>("/portfolio/samples"),
  analyze: (holdings: Holding[]) => post<Analysis>("/portfolio/analyze", { holdings }),
  plan: (goal: GoalInput) => post<GoalPlan>("/goals/plan", goal),
  search: (q: string, category: string) =>
    request<SearchResult>(
      `/kb/search?q=${encodeURIComponent(q)}${category ? `&category=${encodeURIComponent(category)}` : ""}`,
    ),
  sources: () => request<{ chunks: number; sources: Source[] }>("/kb/sources"),
  metrics: () => request<Metrics>("/metrics"),
};

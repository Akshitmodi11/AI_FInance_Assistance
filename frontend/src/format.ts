export const usd = (x: number, digits = 0) =>
  x.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

export const pct = (x: number | null | undefined, digits = 0) =>
  x == null ? "—" : `${(x * 100).toFixed(digits)}%`;

export const signedPct = (x: number, digits = 2) => `${x >= 0 ? "+" : ""}${x.toFixed(digits)}%`;

export const AGENTS: Record<string, { label: string; color: string }> = {
  finance_qa: { label: "Finance Q&A", color: "#4c8df6" },
  tax: { label: "Tax", color: "#a77bf3" },
  portfolio: { label: "Portfolio", color: "#2ebd75" },
  goal: { label: "Goals", color: "#e9b949" },
  market: { label: "Market", color: "#22b8cf" },
  news: { label: "News", color: "#f0864f" },
};

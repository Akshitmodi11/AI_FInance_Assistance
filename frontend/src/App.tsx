import { useEffect, useState } from "react";
import { api } from "./api";
import { Icon, type IconName } from "./components/icons";
import { Sidebar } from "./components/Sidebar";
import { signedPct } from "./format";
import { Chat } from "./pages/Chat";
import { Goals } from "./pages/Goals";
import { Insights } from "./pages/Insights";
import { Knowledge } from "./pages/Knowledge";
import { Markets } from "./pages/Markets";
import { Portfolio } from "./pages/Portfolio";
import type { Analysis, ChatTurn, GoalPlan, Holding, IndexQuote, Profile, Status } from "./types";

const TABS: { id: string; label: string; icon: IconName }[] = [
  { id: "chat", label: "Chat", icon: "chat" },
  { id: "portfolio", label: "Portfolio", icon: "portfolio" },
  { id: "markets", label: "Markets", icon: "markets" },
  { id: "goals", label: "Goals", icon: "goals" },
  { id: "knowledge", label: "Knowledge", icon: "knowledge" },
  { id: "insights", label: "Insights", icon: "insights" },
];

const tabFromHash = () => {
  const id = window.location.hash.slice(1);
  return TABS.some((t) => t.id === id) ? id : "chat";
};

type Theme = "dark" | "light";
function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem("finnie-theme");
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    /* storage blocked */
  }
  return "dark";
}

export default function App() {
  const [tab, setTab] = useState(tabFromHash);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [status, setStatus] = useState<Status | null>(null);
  const [indexes, setIndexes] = useState<IndexQuote[]>([]);
  const [profile, setProfile] = useState<Profile>({ knowledge_level: "beginner", risk_tolerance: "moderate", horizon_years: 10 });
  // Shared across tabs: the chat thread, the analyzed portfolio and the last goal plan.
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [holdings, setHoldings] = useState<Holding[] | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [plan, setPlan] = useState<GoalPlan | null>(null);

  useEffect(() => {
    api.status().then(setStatus).catch(() => {});
    api.indexes().then(setIndexes).catch(() => {});
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("finnie-theme", theme);
    } catch {
      /* storage blocked */
    }
  }, [theme]);

  const go = (id: string) => {
    window.location.hash = id;
    setTab(id);
  };

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 16l5-5 4 4 7-8" /></svg>
          </div>
          <span className="brand-name">Finnie</span>
          <span className="brand-tag">AI finance education</span>
        </div>
        <div className="ticker-strip" aria-label="Market overview">
          {indexes.map((i) => (
            <div key={i.ticker} className="tick" title={`${i.name} — tracked via the ${i.ticker} ETF`}>
              <span className="sym">{i.name} <span className="caption">{i.ticker}</span></span>
              {i.price == null ? (
                <span className="caption">n/a</span>
              ) : (
                <>
                  <span className="px">{i.price.toFixed(2)}</span>
                  <span className={`chg ${(i.change_pct ?? 0) >= 0 ? "up" : "down"}`}>
                    {(i.change_pct ?? 0) >= 0 ? "▲" : "▼"} {signedPct(i.change_pct ?? 0)}
                  </span>
                </>
              )}
            </div>
          ))}
        </div>
        <button
          className="icon-btn"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          <Icon name={theme === "dark" ? "sun" : "moon"} />
        </button>
      </header>

      <div className="body">
        <Sidebar
          profile={profile}
          setProfile={setProfile}
          status={status}
          onNewChat={() => {
            setThreadId(crypto.randomUUID());
            setTurns([]);
            go("chat");
          }}
        />
        <div className="main">
          <nav className="tabs" role="tablist">
            {TABS.map((t) => (
              <button key={t.id} role="tab" aria-selected={tab === t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => go(t.id)}>
                <Icon name={t.icon} /> {t.label}
              </button>
            ))}
          </nav>
          <main className="page" style={tab === "chat" ? { flex: 1 } : undefined}>
            {tab === "chat" && <Chat turns={turns} setTurns={setTurns} threadId={threadId} profile={profile} portfolio={holdings} />}
            {tab === "portfolio" && (
              <Portfolio analysis={analysis} onAnalyzed={(h, a) => { setHoldings(h); setAnalysis(a); }} />
            )}
            {tab === "markets" && <Markets indexes={indexes} />}
            {tab === "goals" && (
              <Goals risk={profile.risk_tolerance} riskOptions={status?.risk_profiles ?? ["conservative", "moderate", "aggressive"]} plan={plan} setPlan={setPlan} />
            )}
            {tab === "knowledge" && <Knowledge status={status} />}
            {tab === "insights" && <Insights status={status} />}
          </main>
        </div>
      </div>
    </div>
  );
}

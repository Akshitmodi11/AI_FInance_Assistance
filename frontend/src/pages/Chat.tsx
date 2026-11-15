import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { Answer } from "../components/Answer";
import { Icon, type IconName } from "../components/icons";
import type { ChatTurn, Holding, Profile } from "../types";

const EXAMPLES: { kind: string; icon: IconName; q: string }[] = [
  { kind: "Tax", icon: "knowledge", q: "What's the difference between a traditional and a Roth IRA?" },
  { kind: "Portfolio", icon: "portfolio", q: "Analyze my portfolio" },
  { kind: "Goals", icon: "goals", q: "I want $40,000 for a house in 6 years. I have $5,000 and can add $400 a month." },
  { kind: "Markets + news", icon: "markets", q: "How is nvidia doing, and any news on the stock market?" },
  { kind: "Guardrails", icon: "shield", q: "Should I buy Tesla?" },
];

export function Chat({
  turns,
  setTurns,
  threadId,
  profile,
  portfolio,
}: {
  turns: ChatTurn[];
  setTurns: (f: (t: ChatTurn[]) => ChatTurn[]) => void;
  threadId: string;
  profile: Profile;
  portfolio: Holding[] | null;
}) {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // braces matter: newer browsers return a Promise from scrollIntoView, and React would try to
    // call an effect's return value as its cleanup function
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, busy]);

  async function send(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    setInput("");
    setError(null);
    setBusy(true);
    setTurns((t) => [...t, { role: "user", content: q }]);
    try {
      const answer = await api.chat(q, threadId, profile, portfolio);
      setTurns((t) => [...t, { role: "assistant", answer }]);
    } catch (e) {
      setTurns((t) => t.slice(0, -1));
      setInput(q);
      setError(`Something went wrong: ${(e as Error).message}. Please try again.`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="chat">
      {turns.length === 0 && (
        <>
          <div className="welcome">
            <h2>What would you like to understand today?</h2>
            <p>
              Ask about investing, taxes, savings goals or markets. Six specialist agents answer with cited sources and
              live data{portfolio ? " — including the portfolio you analyzed" : ""}.
            </p>
          </div>
          <div className="examples">
            {EXAMPLES.map((e) => (
              <button key={e.q} className="example" onClick={() => send(e.q)}>
                <Icon name={e.icon} />
                <span>
                  <span className="kind">{e.kind}</span>
                  {e.q}
                </span>
              </button>
            ))}
          </div>
        </>
      )}

      {turns.map((t, i) => (
        <div key={i} className={`msg ${t.role}`}>
          <div className="avatar">
            {t.role === "user" ? (
              <Icon name="user" />
            ) : (
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 16l5-5 4 4 7-8" /></svg>
            )}
          </div>
          <div className="bubble">{t.role === "user" ? t.content : <Answer a={t.answer} />}</div>
        </div>
      ))}
      {busy && (
        <div className="msg assistant">
          <div className="avatar">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 16l5-5 4 4 7-8" /></svg>
          </div>
          <div className="bubble loading">
            <span className="dots"><span /><span /><span /></span> Routing to the right agents…
          </div>
        </div>
      )}
      {error && (
        <div className="note bad">
          <Icon name="alert" /> <span>{error}</span>
        </div>
      )}
      <div ref={end} />

      <div className="composer">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask Finnie about investing, taxes, goals or a stock…"
            aria-label="Ask Finnie"
            maxLength={2000}
            autoFocus
          />
          <button className="btn primary" disabled={busy || !input.trim()} aria-label="Send">
            <Icon name="send" /> Send
          </button>
        </form>
        <div className="caption hint">Educational only — not financial advice.</div>
      </div>
    </div>
  );
}

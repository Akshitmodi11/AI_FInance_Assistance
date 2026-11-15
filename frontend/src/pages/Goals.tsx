import { useState, type ChangeEvent, type FormEvent } from "react";
import { api } from "../api";
import { GoalChart } from "../components/charts";
import { ErrorNote, PageHead } from "../components/ui";
import { pct, usd } from "../format";
import type { GoalInput, GoalPlan, Risk } from "../types";

export function Goals({
  risk,
  riskOptions,
  plan,
  setPlan,
}: {
  risk: Risk;
  riskOptions: Risk[];
  plan: GoalPlan | null;
  setPlan: (p: GoalPlan) => void;
}) {
  const [form, setForm] = useState({
    goal_name: "House down payment",
    target_amount: "40000",
    years: "6",
    current_savings: "5000",
    monthly_contribution: "400",
    risk_profile: risk,
  });
  const [error, setError] = useState<string | null>(null);
  const set = (k: keyof typeof form) => (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.value });

  async function submit(e: FormEvent) {
    e.preventDefault();
    const goal: GoalInput = {
      goal_name: form.goal_name || "your goal",
      target_amount: Number(form.target_amount),
      years: Number(form.years),
      current_savings: Number(form.current_savings) || 0,
      monthly_contribution: Number(form.monthly_contribution) || null,
      risk_profile: form.risk_profile as Risk,
    };
    try {
      setError(null);
      setPlan(await api.plan(goal));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <>
      <PageHead title="Goal planner" sub="How much to save each month, under low, expected and high return scenarios." />
      <form className="card grid grid-3" onSubmit={submit}>
        <label className="field">Goal<input className="input" value={form.goal_name} onChange={set("goal_name")} /></label>
        <label className="field">Target (today's $)<input className="input" type="number" min={100} step={100} value={form.target_amount} onChange={set("target_amount")} required /></label>
        <label className="field">Years<input className="input" type="number" min={0.5} max={60} step={0.5} value={form.years} onChange={set("years")} required /></label>
        <label className="field">Already saved ($)<input className="input" type="number" min={0} step={100} value={form.current_savings} onChange={set("current_savings")} /></label>
        <label className="field">Monthly saving ($, 0 = tell me what's needed)<input className="input" type="number" min={0} step={50} value={form.monthly_contribution} onChange={set("monthly_contribution")} /></label>
        <label className="field">Risk appetite
          <select className="input" value={form.risk_profile} onChange={set("risk_profile")}>
            {riskOptions.map((r) => <option key={r}>{r}</option>)}
          </select>
        </label>
        <div><button className="btn primary">Plan it</button></div>
      </form>
      <ErrorNote error={error} />

      {plan ? (
        <>
          <p style={{ margin: 0 }}>
            <b>{plan.goal.goal_name}</b>: {usd(plan.goal.target_amount)} today ≈ <b>{usd(plan.target_in_future_dollars)}</b> in{" "}
            {plan.goal.years} years at {pct(plan.inflation, 1)} inflation.
          </p>
          <div className="grid grid-3">
            {plan.scenarios.map((s) => (
              <div key={s.label} className="card metric">
                <span className="label">{s.label[0].toUpperCase() + s.label.slice(1)} return ({pct(s.annual_return)}/yr)</span>
                <span className="value">{usd(s.required_monthly)}/mo needed</span>
                {s.on_track != null && (
                  <span className={`delta ${s.on_track ? "up" : "down"}`}>{s.on_track ? "✓ on track" : "✗ short of target"}</span>
                )}
              </div>
            ))}
          </div>
          <div className="card"><GoalChart plan={plan} /></div>
          <details className="more">
            <summary>Year-by-year projection</summary>
            <table className="table">
              <thead><tr><th>Year</th><th className="num">Low</th><th className="num">Expected</th><th className="num">High</th></tr></thead>
              <tbody>
                {plan.yearly_projection.map((r) => (
                  <tr key={r.year}><td>{r.year}</td><td className="num">{usd(r.low)}</td><td className="num">{usd(r.expected)}</td><td className="num">{usd(r.high)}</td></tr>
                ))}
              </tbody>
            </table>
          </details>
          <div className="caption">
            Return assumptions for <b>{plan.goal.risk_profile}</b> come from config.yaml. Higher expected returns come with bigger
            swings, and no return is guaranteed.
          </div>
        </>
      ) : (
        <div className="caption">Fill in the goal and press <b>Plan it</b>.</div>
      )}
    </>
  );
}

"""Goals tab — savings-goal calculator with risk-appetite return bands and a projection chart."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agents.tools import plan_goal
from src.core.config import get_settings
from src.core.schemas import GoalInput
from src.web_app.components import charts


def render(profile: dict) -> None:
    st.subheader("🎯 Goal planner")
    goals_cfg = get_settings().goals
    risk_options = list(goals_cfg.risk_profiles)

    with st.form("goal"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Goal", "House down payment")
        target = c2.number_input("Target (today's $)", min_value=100.0, value=40_000.0, step=1000.0)
        years = c3.number_input("Years", min_value=0.5, max_value=60.0, value=6.0, step=0.5)
        c4, c5, c6 = st.columns(3)
        current = c4.number_input("Already saved ($)", min_value=0.0, value=5_000.0, step=500.0)
        monthly = c5.number_input(
            "Monthly saving ($, 0 = tell me what's needed)", min_value=0.0, value=400.0, step=50.0
        )
        risk = c6.selectbox(
            "Risk appetite", risk_options, index=risk_options.index(profile["risk_tolerance"])
        )
        submitted = st.form_submit_button("Plan it", type="primary")

    if not submitted and "goal_plan" not in st.session_state:
        st.caption("Fill in the goal and press **Plan it**.")
        return
    if submitted:
        goal = GoalInput(
            goal_name=name,
            target_amount=target,
            years=years,
            current_savings=current,
            monthly_contribution=monthly or None,
            risk_profile=risk,
        )
        st.session_state.goal_plan = plan_goal(goal, goals_cfg).model_dump(mode="json")
    plan = st.session_state.goal_plan
    goal = plan["goal"]

    st.markdown(
        f"**{goal['goal_name']}**: ${goal['target_amount']:,.0f} today ≈ "
        f"**${plan['target_in_future_dollars']:,.0f}** in {goal['years']:g} years at "
        f"{plan['inflation']:.1%} inflation."
    )
    cols = st.columns(3)
    for col, s in zip(cols, plan["scenarios"], strict=True):
        delta = None
        if s["on_track"] is not None:
            delta = "on track" if s["on_track"] else "short of target"
        col.metric(
            f"{s['label'].title()} return ({s['annual_return']:.0%}/yr)",
            f"${s['required_monthly']:,.0f}/mo needed",
            delta,
            delta_color="normal" if s["on_track"] else "inverse",
            border=True,
        )
    st.plotly_chart(charts.goal_projection(plan), key="goal-chart")
    with st.expander("Year-by-year projection"):
        st.dataframe(pd.DataFrame(plan["yearly_projection"]).round(0), hide_index=True)
    st.caption(
        f"Return assumptions for **{goal['risk_profile']}** come from config.yaml. Higher expected "
        "returns come with bigger swings, and no return is guaranteed."
    )

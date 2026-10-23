"""Typed data contracts shared by agents, tools, the graph, the UI and the MCP server.

Everything an agent hands to another component is one of these models, so a wrong shape fails
loudly at the boundary instead of surfacing as a confusing LLM answer later.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Intent = Literal["finance_qa", "tax", "portfolio", "goal", "market", "news"]
INTENTS: tuple[Intent, ...] = ("finance_qa", "tax", "portfolio", "goal", "market", "news")
KnowledgeLevel = Literal["beginner", "intermediate", "advanced"]
RiskProfile = Literal["conservative", "moderate", "aggressive"]
AssetClass = Literal["stock", "etf", "bond", "cash", "crypto", "other"]


class UserProfile(BaseModel):
    knowledge_level: KnowledgeLevel = "beginner"
    risk_tolerance: RiskProfile = "moderate"
    horizon_years: int = Field(10, ge=0, le=80)


# ---------- Portfolio ----------


class Holding(BaseModel):
    ticker: str
    shares: float = Field(gt=0)
    cost_basis: float | None = None  # per share, optional

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class Portfolio(BaseModel):
    holdings: list[Holding] = Field(default_factory=list)


class PositionMetrics(BaseModel):
    ticker: str
    shares: float
    price: float
    value: float
    weight: float
    asset_class: AssetClass
    sector: str
    expense_ratio: float | None = None


class PortfolioAnalysis(BaseModel):
    total_value: float
    positions: list[PositionMetrics]
    hhi: float  # Herfindahl index of weights: 1/n (spread out) … 1.0 (one holding)
    effective_n: float  # "how many equal-sized holdings this behaves like"
    top_holding_weight: float
    asset_class_mix: dict[str, float]
    sector_mix: dict[str, float]
    weighted_expense_ratio: float | None
    annualized_volatility: float | None
    diversification_score: int  # 0–100, for the gauge
    observations: list[str]  # educational, never buy/sell instructions
    data_source: str
    is_mock: bool = False


# ---------- Market data ----------


class Quote(BaseModel):
    ticker: str
    price: float
    change: float
    change_pct: float
    currency: str = "USD"
    as_of: datetime
    source: str
    is_cached: bool = False
    is_mock: bool = False


class PricePoint(BaseModel):
    date: datetime
    close: float


class PriceHistory(BaseModel):
    ticker: str
    points: list[PricePoint]
    source: str
    is_cached: bool = False
    is_mock: bool = False

    @property
    def closes(self) -> list[float]:
        return [p.close for p in self.points]


class MarketSnapshot(BaseModel):
    """Facts computed in Python (never by the LLM) for the Market agent to explain."""

    quote: Quote
    high_52w: float | None
    low_52w: float | None
    sma_20: float | None
    sma_50: float | None
    trend: Literal["uptrend", "downtrend", "sideways", "unknown"]
    return_1m: float | None
    return_6m: float | None


class NewsItem(BaseModel):
    title: str
    url: str
    source: str
    published: str | None = None
    summary: str = ""
    is_mock: bool = False


# ---------- Goals ----------


class GoalInput(BaseModel):
    target_amount: float = Field(gt=0)
    years: float = Field(gt=0, le=80)
    current_savings: float = Field(0, ge=0)
    monthly_contribution: float | None = Field(None, ge=0)  # None → compute what's required
    risk_profile: RiskProfile = "moderate"
    goal_name: str = "your goal"


class GoalScenario(BaseModel):
    label: Literal["low", "expected", "high"]
    annual_return: float
    required_monthly: float  # to hit the (inflation-adjusted) target
    projected_value: float | None  # with the user's monthly contribution, if given
    on_track: bool | None


class GoalPlan(BaseModel):
    goal: GoalInput
    inflation: float
    target_in_future_dollars: float
    scenarios: list[GoalScenario]
    yearly_projection: list[dict[str, float]]  # [{"year", "low", "expected", "high"}] for charts


# ---------- Agents / graph ----------


class Citation(BaseModel):
    label: str  # "IRS Publication 590-A, p. 40"
    url: str
    source: str


class AgentResult(BaseModel):
    agent: Intent
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    data: dict = Field(default_factory=dict)  # structured payload for the UI (charts, tables)
    confident: bool = True
    error: str | None = None


class RouterDecision(BaseModel):
    """Structured output of the router LLM."""

    intents: list[Intent] = Field(min_length=1, max_length=3)
    tickers: list[str] = Field(default_factory=list)
    standalone_question: str = Field(
        description="The user's latest question rewritten to make sense on its own, "
        "resolving follow-ups like 'what about for a Roth?' using the conversation."
    )


class FinalResponse(BaseModel):
    answer: str
    agents: list[Intent]
    citations: list[Citation] = Field(default_factory=list)
    data: dict = Field(default_factory=dict)
    guardrail_note: str | None = None
    disclaimer: str

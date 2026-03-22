"""
Domain models for the Freqtrade Strategy Research Agent.
These are the canonical data structures shared across all modules.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional


# ============================================================
# Enums
# ============================================================

class ComponentCategory(str, Enum):
    INDICATOR = "indicator"
    ENTRY_SIGNAL = "entry_signal"
    EXIT_SIGNAL = "exit_signal"
    FILTER = "filter"
    RISK_RULE = "risk_rule"
    POSITION_SIZING = "position_sizing"
    REGIME_FILTER = "regime_filter"


class ComponentOrigin(str, Enum):
    HUMAN_AUTHORED = "human_authored"
    LLM_GENERATED = "llm_generated"
    MUTATION = "mutation"


class HypothesisType(str, Enum):
    EXPLORE = "EXPLORE"
    EXPLOIT = "EXPLOIT"
    ABLATE = "ABLATE"
    CROSSOVER = "CROSSOVER"
    STRESS_TEST = "STRESS_TEST"
    REGIME_SPECIALIZE = "REGIME_SPECIALIZE"


class HypothesisStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ABANDONED = "abandoned"


class HypothesisOutcome(str, Enum):
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class StrategyStatus(str, Enum):
    CREATED = "created"
    BACKTESTING = "backtesting"
    EVALUATED = "evaluated"
    PROMOTED = "promoted"
    ARCHIVED = "archived"
    FAILED = "failed"


class RegimeTrend(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    UNKNOWN = "unknown"


class RegimeVol(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    UNKNOWN = "unknown"


class InsightType(str, Enum):
    COMPONENT_EFFECT = "component_effect"
    REGIME_CONDITION = "regime_condition"
    PARAMETER_SENSITIVITY = "parameter_sensitivity"
    INTERACTION_EFFECT = "interaction_effect"
    FAILURE_PATTERN = "failure_pattern"
    ROBUSTNESS_SIGNAL = "robustness_signal"


# ============================================================
# Data Models
# ============================================================

@dataclass
class Component:
    id: uuid.UUID
    category: ComponentCategory
    name: str
    code_snippet: str
    parameters: dict[str, Any]
    parameter_space: dict[str, Any]   # {param: [min, max]} or {param: [choices]}
    dependencies: list[str]
    description: str
    origin: ComponentOrigin
    parent_id: Optional[uuid.UUID] = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComponentPerformance:
    id: uuid.UUID
    component_id: uuid.UUID
    pair: str
    timeframe: str
    regime_trend: RegimeTrend
    regime_vol: RegimeVol
    appearances: int = 0
    ablation_count: int = 0
    avg_sharpe_contribution: Optional[float] = None
    avg_win_rate_delta: Optional[float] = None
    avg_drawdown_delta: Optional[float] = None
    confidence: float = 0.0


@dataclass
class Strategy:
    id: uuid.UUID
    name: str
    code: str
    component_ids: list[uuid.UUID]
    parameters: dict[str, Any]
    hypothesis_id: Optional[uuid.UUID] = None
    parent_ids: list[uuid.UUID] = field(default_factory=list)
    generation: int = 0
    status: StrategyStatus = StrategyStatus.CREATED
    best_score: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BacktestMetrics:
    """Parsed results from a single freqtrade backtest run."""
    sharpe: Optional[float]
    sortino: Optional[float]
    profit_factor: Optional[float]
    max_drawdown: Optional[float]
    win_rate: Optional[float]
    trade_count: Optional[int]
    avg_trade_dur_h: Optional[float]
    calmar: Optional[float]
    total_profit_pct: Optional[float]
    monthly_profit_mean: Optional[float]
    monthly_profit_std: Optional[float]


@dataclass
class BacktestRun:
    id: uuid.UUID
    strategy_id: uuid.UUID
    pair: str
    timeframe: str
    date_from: date
    date_to: date
    is_holdout: bool
    regime_trend: RegimeTrend
    regime_vol: RegimeVol
    regime_metadata: dict[str, Any]
    metrics: BacktestMetrics
    oos_sharpe_delta: Optional[float]
    trade_count_stability: Optional[float]
    composite_score: Optional[float]
    disqualified: bool
    disqualification_reason: Optional[str]
    raw_output: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Hypothesis:
    id: uuid.UUID
    type: HypothesisType
    rationale: str
    prediction: str
    priority_score: float
    target_component_ids: list[uuid.UUID]
    target_strategy_id: Optional[uuid.UUID]
    parameter_changes: dict[str, Any]
    context: dict[str, Any]   # {pair, timeframe, regime}
    status: HypothesisStatus = HypothesisStatus.QUEUED
    outcome: Optional[HypothesisOutcome] = None
    resulting_strategy_id: Optional[uuid.UUID] = None
    outcome_summary: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Insight:
    id: uuid.UUID
    statement: str
    insight_type: InsightType
    confidence: float
    evidence_run_ids: list[uuid.UUID]
    contradicted_by: list[uuid.UUID]
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExperimentContext:
    """Runtime context passed through the experiment pipeline."""
    hypothesis: Hypothesis
    strategy: Optional[Strategy] = None
    backtest_run: Optional[BacktestRun] = None
    ablation_run: Optional[BacktestRun] = None

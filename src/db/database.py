"""
Database access layer — async PostgreSQL via asyncpg / SQLAlchemy core.
All DB interactions go through this module.
"""
from __future__ import annotations
import uuid
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

import asyncpg
import structlog

from src.config import settings
from src.models import (
    BacktestRun, Component, ComponentPerformance, Hypothesis,
    HypothesisStatus, HypothesisType, Insight, Strategy, StrategyStatus,
    RegimeTrend, RegimeVol, BacktestMetrics, InsightType
)

log = structlog.get_logger()


class Database:
    """Async database client wrapping asyncpg connection pool."""

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        self._pool = await asyncpg.create_pool(
            settings.postgres_dsn,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
        log.info("database.connected", dsn=settings.postgres_dsn)

    async def disconnect(self):
        if self._pool:
            await self._pool.close()

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        async with self._pool.acquire() as conn:
            yield conn

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    async def save_component(self, c: Component) -> None:
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO components
                    (id, category, name, code_snippet, parameters, parameter_space,
                     dependencies, description, origin, parent_id, tags, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (name) DO UPDATE SET
                    code_snippet = EXCLUDED.code_snippet,
                    parameters = EXCLUDED.parameters,
                    parameter_space = EXCLUDED.parameter_space,
                    description = EXCLUDED.description
                """,
                c.id, c.category.value, c.name, c.code_snippet,
                json.dumps(c.parameters), json.dumps(c.parameter_space),
                c.dependencies, c.description, c.origin.value,
                c.parent_id, c.tags, c.created_at,
            )

    async def get_all_components(self) -> list[dict[str, Any]]:
        async with self.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM components ORDER BY created_at")
            return [dict(r) for r in rows]

    async def get_component_scores(
        self, pair: str = None, timeframe: str = None
    ) -> list[dict[str, Any]]:
        async with self.acquire() as conn:
            query = "SELECT * FROM v_component_leaderboard"
            return [dict(r) for r in await conn.fetch(query)]

    async def update_component_performance(self, cp: ComponentPerformance) -> None:
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO component_performance
                    (id, component_id, pair, timeframe, regime_trend, regime_vol,
                     appearances, ablation_count, avg_sharpe_contribution,
                     avg_win_rate_delta, avg_drawdown_delta, confidence, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,NOW())
                ON CONFLICT (component_id, pair, timeframe, regime_trend, regime_vol)
                DO UPDATE SET
                    appearances = component_performance.appearances + EXCLUDED.appearances,
                    ablation_count = component_performance.ablation_count + EXCLUDED.ablation_count,
                    avg_sharpe_contribution = EXCLUDED.avg_sharpe_contribution,
                    avg_win_rate_delta = EXCLUDED.avg_win_rate_delta,
                    avg_drawdown_delta = EXCLUDED.avg_drawdown_delta,
                    confidence = EXCLUDED.confidence,
                    updated_at = NOW()
                """,
                cp.id, cp.component_id, cp.pair, cp.timeframe,
                cp.regime_trend.value, cp.regime_vol.value,
                cp.appearances, cp.ablation_count,
                cp.avg_sharpe_contribution, cp.avg_win_rate_delta,
                cp.avg_drawdown_delta, cp.confidence,
            )

    # --------------------------------------------------------
    # Strategies
    # --------------------------------------------------------

    async def save_strategy(self, s: Strategy) -> None:
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO strategies
                    (id, name, code, component_ids, parameters, hypothesis_id,
                     parent_ids, generation, status, best_score, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                """,
                s.id, s.name, s.code,
                [str(c) for c in s.component_ids],
                json.dumps(s.parameters),
                s.hypothesis_id,
                [str(p) for p in s.parent_ids],
                s.generation, s.status.value, s.best_score, s.created_at,
            )

    async def update_strategy_status(
        self, strategy_id: uuid.UUID, status: StrategyStatus, best_score: float = None
    ) -> None:
        async with self.acquire() as conn:
            await conn.execute(
                "UPDATE strategies SET status=$1, best_score=COALESCE($2, best_score) WHERE id=$3",
                status.value, best_score, strategy_id,
            )

    async def get_strategy(self, strategy_id: uuid.UUID) -> Optional[dict]:
        async with self.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM strategies WHERE id=$1", strategy_id)
            return dict(row) if row else None

    async def get_top_strategies(
        self, limit: int = 10, pair: str = None, timeframe: str = None
    ) -> list[dict]:
        async with self.acquire() as conn:
            clauses = []
            params: list[Any] = []

            if pair:
                params.append(pair)
                clauses.append(f"pair = ${len(params)}")
            if timeframe:
                params.append(timeframe)
                clauses.append(f"timeframe = ${len(params)}")

            params.append(max(1, int(limit)))
            query = "SELECT * FROM v_best_strategies"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += f" LIMIT ${len(params)}"

            return [dict(r) for r in await conn.fetch(query, *params)]

    # --------------------------------------------------------
    # Backtest Runs
    # --------------------------------------------------------

    async def save_backtest_run(self, run: BacktestRun) -> None:
        m = run.metrics
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO backtest_runs
                    (id, strategy_id, pair, timeframe, date_from, date_to, is_holdout,
                     regime_trend, regime_vol, regime_metadata,
                     sharpe, sortino, profit_factor, max_drawdown, win_rate,
                     trade_count, avg_trade_dur_h, calmar, total_profit_pct,
                     monthly_profit_mean, monthly_profit_std,
                     oos_sharpe_delta, trade_count_stability,
                     composite_score, disqualified, disqualification_reason,
                     raw_output, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,
                        $17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28)
                """,
                run.id, run.strategy_id, run.pair, run.timeframe,
                run.date_from, run.date_to, run.is_holdout,
                run.regime_trend.value, run.regime_vol.value,
                json.dumps(run.regime_metadata),
                m.sharpe, m.sortino, m.profit_factor, m.max_drawdown,
                m.win_rate, m.trade_count, m.avg_trade_dur_h, m.calmar,
                m.total_profit_pct, m.monthly_profit_mean, m.monthly_profit_std,
                run.oos_sharpe_delta, run.trade_count_stability,
                run.composite_score, run.disqualified, run.disqualification_reason,
                json.dumps(run.raw_output), run.created_at,
            )

    async def get_runs_for_strategy(self, strategy_id: uuid.UUID) -> list[dict]:
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM backtest_runs WHERE strategy_id=$1 ORDER BY created_at",
                strategy_id,
            )
            return [dict(r) for r in rows]

    async def count_completed_experiments(self) -> int:
        async with self.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM hypotheses WHERE status='complete'"
            )

    # --------------------------------------------------------
    # Hypotheses
    # --------------------------------------------------------

    async def save_hypothesis(self, h: Hypothesis) -> None:
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO hypotheses
                    (id, type, rationale, prediction, priority_score,
                     target_component_ids, target_strategy_id, parameter_changes,
                     context, status, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                """,
                h.id, h.type.value, h.rationale, h.prediction,
                h.priority_score,
                [str(c) for c in h.target_component_ids],
                h.target_strategy_id,
                json.dumps(h.parameter_changes),
                json.dumps(h.context),
                h.status.value, h.created_at,
            )

    async def update_hypothesis(self, h: Hypothesis) -> None:
        async with self.acquire() as conn:
            await conn.execute(
                """
                UPDATE hypotheses SET
                    status=$1, outcome=$2, resulting_strategy_id=$3,
                    outcome_summary=$4, started_at=$5, completed_at=$6
                WHERE id=$7
                """,
                h.status.value,
                h.outcome.value if h.outcome else None,
                h.resulting_strategy_id,
                h.outcome_summary,
                h.started_at, h.completed_at, h.id,
            )

    async def get_next_hypothesis(self) -> Optional[dict]:
        """Pop the highest-priority queued hypothesis."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM hypotheses
                WHERE status='queued'
                ORDER BY priority_score DESC
                LIMIT 1
                """
            )
            return dict(row) if row else None

    async def get_recent_hypotheses(self, limit: int = 50) -> list[dict]:
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM hypotheses ORDER BY created_at DESC LIMIT $1", limit
            )
            return [dict(r) for r in rows]

    async def count_recent_by_type(
        self, hypothesis_type: HypothesisType, n: int = 5
    ) -> int:
        async with self.acquire() as conn:
            return await conn.fetchval(
                """
                SELECT COUNT(*) FROM (
                    SELECT id FROM hypotheses
                    WHERE type=$1
                    ORDER BY created_at DESC
                    LIMIT $2
                ) sub
                """,
                hypothesis_type.value, n,
            )

    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    async def save_insight(self, insight: Insight) -> None:
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO insights
                    (id, statement, insight_type, confidence,
                     evidence_run_ids, contradicted_by, active, created_at, last_updated)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                """,
                insight.id, insight.statement, insight.insight_type.value,
                insight.confidence,
                [str(r) for r in insight.evidence_run_ids],
                [str(r) for r in insight.contradicted_by],
                insight.active, insight.created_at, insight.last_updated,
            )

    async def get_active_insights(self) -> list[dict]:
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM insights WHERE active=TRUE ORDER BY confidence DESC"
            )
            return [dict(r) for r in rows]

    async def update_insight_confidence(
        self, insight_id: uuid.UUID, confidence: float,
        contradicted_by: list[uuid.UUID] = None
    ) -> None:
        async with self.acquire() as conn:
            await conn.execute(
                """
                UPDATE insights SET
                    confidence=$1,
                    active = (confidence > 0.15),
                    last_updated=NOW(),
                    contradicted_by = COALESCE($2, contradicted_by)
                WHERE id=$3
                """,
                confidence,
                [str(r) for r in contradicted_by] if contradicted_by else None,
                insight_id,
            )

    # --------------------------------------------------------
    # Knowledge base summary (for hypothesis engine)
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Agent config (persisted on first run, guards holdout dates)
    # --------------------------------------------------------

    async def store_config(self, config: dict[str, Any]) -> None:
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_config (key, value)
                SELECT key, value FROM jsonb_each_text($1::jsonb)
                ON CONFLICT (key) DO NOTHING
                """,
                json.dumps(config),
            )

    # --------------------------------------------------------
    # Failure directives (structured constraints for the engine)
    # --------------------------------------------------------

    async def save_failure_directive(self, directive: dict[str, Any]) -> None:
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO failure_directives
                    (id, constraint_text, root_cause, scope, pair, timeframe,
                     indicators_involved, confidence, evidence_insight_id, created_at)
                VALUES (
                    uuid_generate_v4(), $1, $2, $3, $4, $5, $6, $7, $8, NOW()
                )
                """,
                directive["constraint"],
                directive.get("root_cause", "unknown"),
                directive.get("scope", "general"),
                directive.get("pair"),
                directive.get("timeframe"),
                directive.get("indicators_involved", []),
                directive.get("confidence", 0.6),
                directive.get("evidence_insight_id"),
            )

    async def get_failure_directives(
        self,
        pair: str = None,
        timeframe: str = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get active failure directives — optionally filtered to a context.
        Returns general directives + context-specific ones for the given pair/tf.
        """
        async with self.acquire() as conn:
            if pair and timeframe:
                rows = await conn.fetch(
                    """
                    SELECT * FROM failure_directives
                    WHERE (scope = 'general')
                       OR (scope = 'context_specific' AND pair = $1 AND timeframe = $2)
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT $3
                    """,
                    pair, timeframe, limit,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM failure_directives ORDER BY confidence DESC LIMIT $1",
                    limit,
                )
            return [dict(r) for r in rows]

    async def get_recent_failures_for_context(
        self, pair: str, timeframe: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        """Recent failed backtest runs for a specific context — for Optuna prompt enrichment."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    br.sharpe, br.max_drawdown, br.win_rate, br.trade_count,
                    br.disqualification_reason, br.oos_sharpe_delta,
                    br.trade_count_stability, s.name as strategy_name,
                    s.parameters
                FROM backtest_runs br
                JOIN strategies s ON s.id = br.strategy_id
                WHERE br.pair = $1
                  AND br.timeframe = $2
                  AND (br.disqualified = TRUE OR br.composite_score < 0)
                  AND br.is_holdout = FALSE
                ORDER BY br.created_at DESC
                LIMIT $3
                """,
                pair, timeframe, limit,
            )
            return [dict(r) for r in rows]

    async def get_strategies_fitness_weighted(
        self, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Return top strategies with their fitness score for weighted selection.
        Used by crossover to implement fitness-proportionate parent selection.
        """
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (s.id)
                    s.id as strategy_id, s.name, s.component_ids, s.parameters,
                    s.generation, s.code,
                    b.pair, b.timeframe,
                    b.composite_score, b.sharpe, b.max_drawdown, b.win_rate
                FROM strategies s
                JOIN backtest_runs b ON b.strategy_id = s.id
                WHERE b.disqualified = FALSE
                  AND b.is_holdout = FALSE
                  AND b.composite_score IS NOT NULL
                  AND b.composite_score > 0
                ORDER BY s.id, b.composite_score DESC
                LIMIT $1
                """,
                limit,
            )
            return [dict(r) for r in rows]

    async def get_engine_directives(self) -> list[dict[str, Any]]:
        """
        Engine-level priority directives produced by insight synthesis.
        These bias the hypothesis type sampling in the engine.
        """
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM engine_directives
                WHERE active = TRUE
                ORDER BY created_at DESC
                LIMIT 10
                """
            )
            return [dict(r) for r in rows]

    async def save_engine_directives(self, directives: list[dict[str, Any]]) -> None:
        async with self.acquire() as conn:
            # Deactivate old directives first
            await conn.execute("UPDATE engine_directives SET active = FALSE")
            for d in directives:
                await conn.execute(
                    """
                    INSERT INTO engine_directives
                        (id, directive_type, target_hypothesis_type, rationale,
                         bias_weight, active, created_at)
                    VALUES (uuid_generate_v4(), $1, $2, $3, $4, TRUE, NOW())
                    """,
                    d.get("directive_type", "boost"),
                    d["target_hypothesis_type"],
                    d.get("rationale", ""),
                    float(d.get("bias_weight", 0.2)),
                )

    async def get_stored_config(self) -> Optional[dict[str, Any]]:
        async with self.acquire() as conn:
            rows = await conn.fetch("SELECT key, value FROM agent_config")
            if not rows:
                return None
            return {r["key"]: r["value"] for r in rows}

    async def get_knowledge_summary(self) -> dict[str, Any]:
        """Aggregated snapshot for the hypothesis engine."""
        async with self.acquire() as conn:
            total_runs = await conn.fetchval("SELECT COUNT(*) FROM backtest_runs")
            total_strategies = await conn.fetchval("SELECT COUNT(*) FROM strategies")
            top_components = await conn.fetch(
                "SELECT name, avg_sharpe_contrib, total_appearances FROM v_component_leaderboard LIMIT 20"
            )
            active_insights = await conn.fetch(
                "SELECT statement, confidence, insight_type FROM insights WHERE active=TRUE ORDER BY confidence DESC LIMIT 30"
            )
            coverage = await conn.fetch(
                "SELECT pair, timeframe, COUNT(*) as runs FROM backtest_runs GROUP BY pair, timeframe"
            )
            return {
                "total_runs": total_runs,
                "total_strategies": total_strategies,
                "top_components": [dict(r) for r in top_components],
                "active_insights": [dict(r) for r in active_insights],
                "coverage": [dict(r) for r in coverage],
            }


# Singleton
db = Database()

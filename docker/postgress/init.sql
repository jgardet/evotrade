-- ============================================================
-- Freqtrade Strategy Research Agent — Database Schema
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- for text search on insights

-- ============================================================
-- AGENT CONFIG (persisted on first run to guard holdout dates)
-- ============================================================

CREATE TABLE agent_config (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- COMPONENT LIBRARY
-- ============================================================

CREATE TABLE components (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    category        TEXT NOT NULL CHECK (category IN (
                        'indicator', 'entry_signal', 'exit_signal',
                        'filter', 'risk_rule', 'position_sizing', 'regime_filter'
                    )),
    name            TEXT NOT NULL UNIQUE,
    code_snippet    TEXT NOT NULL,
    parameters      JSONB NOT NULL DEFAULT '{}',
    parameter_space JSONB NOT NULL DEFAULT '{}',  -- {param: [min, max]} or {param: [choices]}
    dependencies    TEXT[] NOT NULL DEFAULT '{}', -- required TA-lib / pandas-ta functions
    description     TEXT,
    origin          TEXT NOT NULL CHECK (origin IN ('human_authored', 'llm_generated', 'mutation')),
    parent_id       UUID REFERENCES components(id),
    tags            TEXT[] DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE component_performance (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component_id            UUID NOT NULL REFERENCES components(id),
    pair                    TEXT NOT NULL,
    timeframe               TEXT NOT NULL,
    regime_trend            TEXT CHECK (regime_trend IN ('trending_up', 'trending_down', 'ranging', 'unknown')),
    regime_vol              TEXT CHECK (regime_vol IN ('low', 'normal', 'high', 'unknown')),
    appearances             INT NOT NULL DEFAULT 0,
    ablation_count          INT NOT NULL DEFAULT 0,   -- how many times ablated
    avg_sharpe_contribution FLOAT,                    -- filled after ablation runs
    avg_win_rate_delta      FLOAT,
    avg_drawdown_delta      FLOAT,
    confidence              FLOAT NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0 AND 1),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (component_id, pair, timeframe, regime_trend, regime_vol)
);

-- ============================================================
-- STRATEGY REGISTRY
-- ============================================================

CREATE TABLE strategies (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,
    code            TEXT NOT NULL,
    component_ids   UUID[] NOT NULL DEFAULT '{}',
    parameters      JSONB NOT NULL DEFAULT '{}',
    hypothesis_id   UUID,                            -- FK added after hypotheses table
    parent_ids      UUID[] NOT NULL DEFAULT '{}',    -- multi-parent for crossover
    generation      INT NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'created' CHECK (status IN (
                        'created', 'backtesting', 'evaluated', 'promoted', 'archived', 'failed'
                    )),
    best_score      FLOAT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE backtest_runs (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id             UUID NOT NULL REFERENCES strategies(id),
    pair                    TEXT NOT NULL,
    timeframe               TEXT NOT NULL,
    date_from               DATE NOT NULL,
    date_to                 DATE NOT NULL,
    is_holdout              BOOLEAN NOT NULL DEFAULT FALSE,

    -- Market regime (computed over the date range)
    regime_trend            TEXT,
    regime_vol              TEXT,
    regime_metadata         JSONB DEFAULT '{}',

    -- Core metrics
    sharpe                  FLOAT,
    sortino                 FLOAT,
    profit_factor           FLOAT,
    max_drawdown            FLOAT,
    win_rate                FLOAT,
    trade_count             INT,
    avg_trade_dur_h         FLOAT,
    calmar                  FLOAT,
    total_profit_pct        FLOAT,
    monthly_profit_mean     FLOAT,
    monthly_profit_std      FLOAT,

    -- Overfitting signals
    oos_sharpe_delta        FLOAT,          -- in_sample_sharpe - holdout_sharpe
    trade_count_stability   FLOAT,          -- coefficient of variation of monthly trades

    -- Composite score (the engine's decision variable)
    composite_score         FLOAT,
    disqualified            BOOLEAN DEFAULT FALSE,
    disqualification_reason TEXT,

    raw_output              JSONB,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- HYPOTHESIS ENGINE
-- ============================================================

CREATE TABLE hypotheses (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type                    TEXT NOT NULL CHECK (type IN (
                                'EXPLORE', 'EXPLOIT', 'ABLATE',
                                'CROSSOVER', 'STRESS_TEST', 'REGIME_SPECIALIZE'
                            )),
    rationale               TEXT NOT NULL,
    prediction              TEXT,
    priority_score          FLOAT NOT NULL DEFAULT 0.0,

    -- What the experiment manipulates
    target_component_ids    UUID[] DEFAULT '{}',
    target_strategy_id      UUID REFERENCES strategies(id),
    parameter_changes       JSONB DEFAULT '{}',
    context                 JSONB DEFAULT '{}',  -- {pair, timeframe, regime}

    status                  TEXT NOT NULL DEFAULT 'queued' CHECK (status IN (
                                'queued', 'running', 'complete', 'abandoned'
                            )),
    outcome                 TEXT CHECK (outcome IN ('confirmed', 'refuted', 'inconclusive')),
    resulting_strategy_id   UUID REFERENCES strategies(id),
    outcome_summary         TEXT,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ
);

-- Add FK from strategies back to hypotheses
ALTER TABLE strategies ADD CONSTRAINT fk_hypothesis
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id);

-- ============================================================
-- INSIGHT STORE
-- ============================================================

CREATE TABLE insights (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    statement           TEXT NOT NULL,
    insight_type        TEXT NOT NULL CHECK (insight_type IN (
                            'component_effect', 'regime_condition',
                            'parameter_sensitivity', 'interaction_effect',
                            'failure_pattern', 'robustness_signal'
                        )),
    confidence          FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    evidence_run_ids    UUID[] NOT NULL DEFAULT '{}',
    contradicted_by     UUID[] NOT NULL DEFAULT '{}',
    active              BOOLEAN NOT NULL DEFAULT TRUE,  -- FALSE = archived
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- FAILURE DIRECTIVES
-- Structured constraints extracted from REFUTED experiments.
-- Consumed by the Hypothesis Engine and Strategy Generator.
-- ============================================================

CREATE TABLE failure_directives (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    constraint_text     TEXT NOT NULL,       -- "Avoid X when Y" — for LLM prompts
    root_cause          TEXT NOT NULL,       -- insufficient_trades, high_drawdown, etc.
    scope               TEXT NOT NULL CHECK (scope IN ('general', 'context_specific')),
    pair                TEXT,                -- only set if context_specific
    timeframe           TEXT,                -- only set if context_specific
    indicators_involved TEXT[] DEFAULT '{}',
    confidence          FLOAT NOT NULL DEFAULT 0.6,
    evidence_insight_id TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_failure_directives_scope ON failure_directives(scope, pair, timeframe);
CREATE INDEX idx_failure_directives_indicators ON failure_directives USING gin(indicators_involved);

-- ============================================================
-- ENGINE DIRECTIVES
-- High-level sampling bias instructions produced by insight synthesis.
-- Tell the engine to boost/suppress specific experiment types.
-- ============================================================

CREATE TABLE engine_directives (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    directive_type          TEXT NOT NULL CHECK (directive_type IN ('boost', 'suppress')),
    target_hypothesis_type  TEXT NOT NULL,   -- which HypothesisType to affect
    rationale               TEXT NOT NULL,
    bias_weight             FLOAT NOT NULL DEFAULT 0.2,  -- how much to shift the weight
    active                  BOOLEAN NOT NULL DEFAULT TRUE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- EXPERIMENT LOG (audit trail)
-- ============================================================

CREATE TABLE experiment_log (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hypothesis_id   UUID REFERENCES hypotheses(id),
    strategy_id     UUID REFERENCES strategies(id),
    event           TEXT NOT NULL,
    details         JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX idx_backtest_strategy_id ON backtest_runs(strategy_id);
CREATE INDEX idx_backtest_composite_score ON backtest_runs(composite_score DESC NULLS LAST);
CREATE INDEX idx_backtest_pair_timeframe ON backtest_runs(pair, timeframe);
CREATE INDEX idx_component_perf_component ON component_performance(component_id);
CREATE INDEX idx_hypotheses_status ON hypotheses(status, priority_score DESC);
CREATE INDEX idx_strategies_status ON strategies(status);
CREATE INDEX idx_insights_active ON insights(active, confidence DESC);
CREATE INDEX idx_insights_text ON insights USING gin(statement gin_trgm_ops);

-- ============================================================
-- VIEWS
-- ============================================================

-- Best strategy per pair/timeframe
CREATE VIEW v_best_strategies AS
SELECT DISTINCT ON (b.pair, b.timeframe)
    s.id AS strategy_id,
    s.name,
    b.pair,
    b.timeframe,
    b.composite_score,
    b.sharpe,
    b.max_drawdown,
    b.trade_count,
    b.win_rate,
    b.regime_trend,
    b.regime_vol
FROM backtest_runs b
JOIN strategies s ON s.id = b.strategy_id
WHERE b.disqualified = FALSE
  AND b.is_holdout = FALSE
ORDER BY b.pair, b.timeframe, b.composite_score DESC;

-- Component leaderboard
CREATE VIEW v_component_leaderboard AS
SELECT
    c.id,
    c.name,
    c.category,
    AVG(cp.avg_sharpe_contribution) AS avg_sharpe_contrib,
    SUM(cp.appearances) AS total_appearances,
    SUM(cp.ablation_count) AS total_ablations,
    AVG(cp.confidence) AS avg_confidence
FROM components c
JOIN component_performance cp ON cp.component_id = c.id
GROUP BY c.id, c.name, c.category
ORDER BY avg_sharpe_contrib DESC NULLS LAST;

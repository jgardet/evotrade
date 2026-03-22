# EvoTrade — Autonomous Trading Strategy Research Agent

Most trading strategy research is a manual grind: tweak an indicator, run a backtest, stare at the results, repeat. EvoTrade automates that entire loop.
It is an autonomous agent that generates Freqtrade trading strategies using an LLM, backtests them against real market data, scores and stores the results, and feeds every insight — including failures — back into the next round of hypothesis generation. The system gets more informed with every experiment it runs.
Under the hood, a Bayesian hypothesis engine decides what to test and why, choosing between exploration of new ideas, exploitation of known-good patterns, crossover of successful components, ablation to isolate what actually drives performance, and stress-testing across market regimes. An LLM then generates the strategy code, Freqtrade runs the backtest, and a second LLM pass periodically synthesizes generalizable patterns from the accumulated knowledge base.
The result is a self-improving research pipeline that runs continuously, learns from its own history, and surfaces insights that would take a human researcher weeks to accumulate.

## Architecture

```
┌─────────────────────────────────────────┐
│           Hypothesis Engine             │
│  (decides what to test and why)         │
└──────────────┬──────────────────────────┘
               ↓
       Strategy Generator (LLM)
       Claude API or OpenAI Codex
               ↓
       Backtest Runner (Freqtrade)
      ┌─────────────────────┐
      │ Auto-download data  │  ← On missing OHLCV
      └─────────────────────┘
               ↓
       Evaluator / Scorer
               ↓
   ┌───────────┴──────────────┐
   │                          │
 Strategy Registry      Component Library
 (PostgreSQL)           (scored, ablated)
   │                          │
   └───────────┬──────────────┘
               ↓
       Insight Synthesis (LLM)
               ↑ loop ↑
```

## Components

- **Hypothesis Engine** — Bayesian prioritization of experiments (EXPLORE, EXPLOIT, ABLATE, CROSSOVER, STRESS-TEST, REGIME-SPECIALIZE)
- **Strategy Generator** — LLM-powered Python strategy code generation
- **Backtest Runner** — Freqtrade CLI wrapper with auto-data-download and artifact fallback
- **Knowledge Base** — PostgreSQL with strategy registry, component library, and insight store
- **Insight Synthesizer** — Periodic LLM pass to extract generalizable patterns
- **Failure Analyzer** — Extracts structured failure patterns from refuted experiments

## Quick Start

```bash
# 1. Copy and configure environment
cp config/env.example .env
# Edit .env with your API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)

# 2. Start infrastructure
docker compose up -d --build

# 3. Run one experiment cycle
docker compose exec -T agent python -m src.agent.main --mode run

# 4. Or run continuously
docker compose exec -T agent python -m src.agent.main --mode loop --max-experiments 100
```

## LLM Backend

Set `LLM_BACKEND` in `.env`:

- `LLM_BACKEND=codex` — Uses OpenAI API (supports `gpt-4o`, `gpt-5.4-nano`, etc.)
- `LLM_BACKEND=claude` — Uses Anthropic Claude (requires `ANTHROPIC_API_KEY`)

### OpenAI Model Compatibility

The client automatically handles parameter compatibility:
- Uses `max_completion_tokens` for newer models
- Falls back to `max_tokens` for older models

## Experiment Modes

```bash
# Run single experiment
docker compose exec -T agent python -m src.agent.main --mode run

# Run continuously
docker compose exec -T agent python -m src.agent.main --mode loop --max-experiments 100

# Force specific hypothesis type
docker compose exec -T agent python -m src.agent.main --mode run --hypothesis-type EXPLORE

# Synthesize insights
docker compose exec -T agent python -m src.agent.main --mode synthesize
```

## Dashboard

View real-time progress at: http://localhost:8501

## Project Structure

```
evotrade/
├── src/
│   ├── agent/           # Main orchestrator
│   ├── backtest/        # Freqtrade runner with auto-fallbacks
│   ├── config.py        # Settings (pydantic-settings)
│   ├── db/              # PostgreSQL access layer
│   ├── hypothesis/      # Engine + synthesizer
│   ├── llm/             # Client with retry + param compatibility
│   ├── models/          # Dataclasses
│   └── strategies/      # Generator
├── config/
│   ├── freqtrade.json   # Freqtrade configuration
│   └── env.example      # Template
├── docker/
│   ├── Dockerfile.agent
│   └── Dockerfile.dashboard
├── docker-compose.yml
└── README.md
```

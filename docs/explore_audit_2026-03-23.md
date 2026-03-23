# EXPLORE Audit — 2026-03-23

## Executive Summary

The EXPLORE pipeline is failing for two different reasons:

1. **Bootstrap EXPLORE runs are unconstrained and fragile.**
   When the component library has fewer than two components, the system generates full strategies directly from a broad LLM prompt with `n_components=0`, which leads to runtime errors, zero-trade strategies, and overly restrictive entry logic.
2. **Runtime repair is not reliably executable.**
   Repaired strategies were saved under a new filename such as `*_retry1`, but the class name inside the Python source was not normalized to match the strategy name that Freqtrade loads.
3. **Bootstrap context selection is sticky.**
   Early EXPLORE hypotheses defaulted to the first configured context instead of the least-covered context, so failures clustered on `BTC/USDT 1h`.
4. **Observability shows the symptoms clearly.**
   The logs show repeated `n_components=0`, frequent `Insufficient trades`, high drawdown disqualifications, one Bollinger-band runtime error, and a retry class-loading failure.

## Findings

### 1. Zero-component bootstrap is the main failure mode

- `HypothesisEngine._gen_explore()` falls back to `_bootstrap_explore()` when the component library has fewer than two components.
- `_bootstrap_explore()` previously chose a generic first-principles EXPLORE run with no components.
- `StrategyGenerator.generate_from_hypothesis()` then invoked the LLM with `n_components=0`, forcing the model to invent the entire strategy from scratch.

This is the highest-risk path in the current design because it skips the normal compositional structure that the rest of the system expects.

### 2. The repair loop had a name-resolution bug

- The agent creates a repaired strategy with a new strategy name like `Agent_EXPLORE_..._retry1`.
- Freqtrade loads strategies by class name matching the requested strategy name.
- The repair prompt asked the model to “fix” the strategy, but there was no deterministic post-processing to rename the class to the new strategy name.

This explains the observed error:

- `Impossible to load Strategy 'Agent_EXPLORE_..._retry1'. This class does not exist ...`

### 3. Runtime-safety checks were too weak

- There was no Python syntax validation after generation or repair.
- There was no deterministic normalization step to align the generated class name with the stored strategy name.
- The strategy prompt allowed indicator implementations that are easy for LLMs to misuse, such as Bollinger Band helpers with inconsistent return types.

### 4. EXPLORE is over-indexed on “novelty” before “viability”

The current EXPLORE path rewards novelty and context coverage, but the system does not sufficiently force:

- a minimum viable trade frequency,
- simpler bootstrap logic,
- safer indicator APIs,
- and deterministic starter templates before free-form invention.

## Implemented Fixes

### A. Deterministic bootstrap EXPLORE templates

Bootstrap EXPLORE no longer relies on unconstrained full-strategy synthesis when there are no components. Instead, the generator now creates one of three deterministic, runnable seed strategies:

- EMA pullback
- Donchian/ADX breakout
- Bollinger mean reversion with trend bias

These templates are intentionally:

- long-only,
- limited in condition count,
- equipped with basic protections,
- and designed to generate actual trades so the knowledge base can bootstrap component learning.

### B. Deterministic strategy finalization

All generated and repaired strategies now pass through a finalization step that:

- normalizes the strategy class name to the persisted strategy name,
- applies loss-protection guards,
- and validates the result with Python `ast.parse`.

This closes the repair-loop filename/class mismatch and catches malformed code earlier.

### C. Better bootstrap context selection

Bootstrap EXPLORE now selects the **least-covered configured context** instead of always picking the first configured pair/timeframe.

This should spread early exploration across the configured search space instead of repeatedly concentrating failures on a single context.

### D. UTC deprecation cleanup

The agent and hypothesis engine now use timezone-aware UTC timestamps rather than `datetime.utcnow()`.

## Recommended Next Enhancements

### 1. Persist starter components after the first viable bootstrap runs

The next structural improvement should be to convert deterministic bootstrap templates into reusable components once they prove viable. That would let EXPLORE transition from:

- **free-form bootstrap** → **component compositional search**

more quickly and with lower variance.

### 2. Add a viability gate before backtesting

Add a lightweight preflight validator that rejects strategies before Freqtrade execution if they:

- omit mandatory methods,
- reference unsupported imports,
- use too many conjunctive entry gates,
- or create impossible threshold combinations.

### 3. Make EXPLORE failure-aware in type selection

If the recent EXPLORE window is dominated by:

- `Insufficient trades`,
- runtime failures,
- or extreme drawdowns,

the engine should automatically:

- downweight free-form EXPLORE,
- upweight ABLATE/STRESS_TEST on surviving seeds,
- or switch to “bootstrap-safe EXPLORE” mode explicitly.

### 4. Add post-backtest signal diagnostics

Capture extra telemetry from candidate strategies:

- number of raw entry signals,
- percent of candles passing each entry clause,
- average hold time,
- and stoploss vs exit-signal exit counts.

This would make `failure_analyzer` far more actionable for “0 trades” and “high drawdown” cases.

### 5. Add component seeding and extraction

The system currently has component mutation/generation support, but bootstrap does not appear to automatically populate a starter library. Seeding a small curated component set would reduce dependence on full-class generation.

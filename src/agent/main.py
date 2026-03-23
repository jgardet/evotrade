"""
Main Agent Loop — orchestrates the full experiment pipeline.

Flow per cycle:
  1. Hypothesis Engine selects what to test
  2. Strategy Generator writes the code
  3. Backtest Runner executes (in-sample + holdout)
  4. Results scored and saved to KB
  5. Component scores updated
  6. Every N runs: Insight Synthesis pass
  7. Repeat
"""
from __future__ import annotations
import argparse
import asyncio
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import structlog

from src.config import settings
from src.db.database import db
from src.models import (
    Component,
    ComponentCategory,
    ComponentOrigin,
    HypothesisOutcome, HypothesisStatus, HypothesisType,
    Strategy,
    StrategyStatus, ComponentPerformance, RegimeTrend, RegimeVol,
)
from src.hypothesis.engine import engine
from src.hypothesis.synthesizer import synthesizer
from src.hypothesis.failure_analyzer import analyzer as failure_analyzer
from src.llm.client import llm
from src.strategies.generator import generator
from src.backtest.runner import runner

log = structlog.get_logger()


class AgentLoop:

    def __init__(self):
        self.experiments_run = 0

    async def startup_checks(self) -> None:
        """
        Run once at startup to validate config and guard the holdout window.
        Raises if the holdout dates have changed since the first run —
        which would silently invalidate all prior out-of-sample results.
        """
        log.info(
            "agent.startup",
            pairs=settings.backtest_pairs,
            timeframes=settings.backtest_timeframes,
            in_sample=f"{settings.in_sample_start} → {settings.in_sample_end}",
            holdout=f"{settings.holdout_start} → {settings.holdout_end}",
            walk_forward=settings.walk_forward_enabled,
            wf_windows=len(settings.walk_forward_windows),
        )

        # Holdout integrity guard — compare stored config against current settings
        stored = await db.get_stored_config()
        if stored:
            stored_holdout_start = stored.get("holdout_start")
            stored_holdout_end = stored.get("holdout_end")
            current_start = str(settings.holdout_start)
            current_end = str(settings.holdout_end)

            if stored_holdout_start and stored_holdout_start != current_start:
                raise RuntimeError(
                    f"HOLDOUT DATE CHANGED: holdout_start was {stored_holdout_start}, "
                    f"now {current_start}. This invalidates your OOS results. "
                    f"Revert the date or reset the database."
                )
            if stored_holdout_end and stored_holdout_end != current_end:
                raise RuntimeError(
                    f"HOLDOUT DATE CHANGED: holdout_end was {stored_holdout_end}, "
                    f"now {current_end}. This invalidates your OOS results. "
                    f"Revert the date or reset the database."
                )
        else:
            # First run — persist the config as the canonical reference
            await db.store_config({
                "holdout_start": str(settings.holdout_start),
                "holdout_end": str(settings.holdout_end),
                "in_sample_start": str(settings.in_sample_start),
                "in_sample_end": str(settings.in_sample_end),
                "backtest_pairs": settings.backtest_pairs,
                "backtest_timeframes": settings.backtest_timeframes,
            })
            log.info("agent.config_stored", msg="First run — configuration persisted to DB")

    async def run_once(self, force_type: str = None) -> bool:
        """Run a single experiment cycle. Returns True if an experiment was run."""

        # 1. Get next hypothesis (or generate one)
        hypothesis_row = await db.get_next_hypothesis()

        if not hypothesis_row:
            log.info("agent.generating_hypothesis")
            hypothesis = await engine.generate_next(force_type=force_type)
            if not hypothesis:
                log.warning("agent.no_hypothesis_available")
                return False
        else:
            hypothesis = self._row_to_hypothesis(hypothesis_row)

        log.info(
            "agent.experiment_start",
            hypothesis_id=str(hypothesis.id),
            type=hypothesis.type.value,
            pair=hypothesis.context.get("pair"),
            timeframe=hypothesis.context.get("timeframe"),
        )

        # Mark as running
        hypothesis.status = HypothesisStatus.RUNNING
        hypothesis.started_at = datetime.now(timezone.utc)
        await db.update_hypothesis(hypothesis)

        try:
            outcome, summary, strategy_id = await self._run_experiment(hypothesis)
        except Exception as e:
            log.error("agent.experiment_error", error=str(e), exc_info=True)
            await engine.close(hypothesis, HypothesisOutcome.INCONCLUSIVE, str(e))
            return True

        await engine.close(hypothesis, outcome, summary, strategy_id)

        self.experiments_run += 1

        # Periodic insight synthesis
        total = await db.count_completed_experiments()
        if total > 0 and total % settings.insight_synthesis_every == 0:
            log.info("agent.insight_synthesis", total_experiments=total)
            await synthesizer.synthesize(recent_n=settings.insight_synthesis_every)

        return True

    async def run_loop(self, max_experiments: int = None):
        """Run continuously until stopped or max_experiments reached."""
        await self.startup_checks()
        log.info("agent.loop_starting", max_experiments=max_experiments)
        while True:
            ran = await self.run_once()
            if not ran:
                log.info("agent.idle_waiting", seconds=30)
                await asyncio.sleep(30)
                continue
            if max_experiments and self.experiments_run >= max_experiments:
                log.info("agent.max_experiments_reached", n=max_experiments)
                break

    # --------------------------------------------------------
    # Experiment Dispatcher
    # --------------------------------------------------------

    async def _run_experiment(
        self, hypothesis
    ) -> tuple[HypothesisOutcome, str, Optional[uuid.UUID]]:
        """
        Dispatch to the appropriate experiment handler based on hypothesis type.
        Returns (outcome, summary, resulting_strategy_id).
        """
        dispatch = {
            HypothesisType.EXPLORE: self._run_explore,
            HypothesisType.EXPLOIT: self._run_exploit,
            HypothesisType.ABLATE: self._run_ablate,
            HypothesisType.CROSSOVER: self._run_crossover,
            HypothesisType.STRESS_TEST: self._run_stress_test,
            HypothesisType.REGIME_SPECIALIZE: self._run_regime_specialize,
        }
        return await dispatch[hypothesis.type](hypothesis)

    async def _run_explore(self, hypothesis):
        """Generate and test a novel strategy."""
        components = await self._load_components(hypothesis.target_component_ids)
        insights = await db.get_active_insights()

        # Generate strategy code
        strategy = await generator.generate_from_hypothesis(hypothesis, components, insights)
        strategy.generation = 0
        await db.save_strategy(strategy)
        await db.update_strategy_status(strategy.id, StrategyStatus.BACKTESTING)

        context = hypothesis.context
        pair = context.get("pair", settings.backtest_pairs[0])
        timeframe = context.get("timeframe", settings.backtest_timeframes[0])

        # --- In-sample run ---
        run = await runner.run(
            strategy, pair, timeframe,
            settings.in_sample_start, settings.in_sample_end,
        )
        await db.save_backtest_run(run)

        if run.disqualified and self._is_runtime_disqualification(run.disqualification_reason):
            try:
                repaired_code = await generator.repair_strategy_code(
                    strategy_name=strategy.name,
                    broken_code=strategy.code,
                    runtime_error=run.disqualification_reason or "",
                    context=context,
                    insights=insights,
                )
                repaired_strategy = Strategy(
                    id=uuid.uuid4(),
                    name=f"{strategy.name}_retry1",
                    code=repaired_code,
                    component_ids=list(strategy.component_ids),
                    parameters=dict(strategy.parameters),
                    hypothesis_id=hypothesis.id,
                    parent_ids=[strategy.id],
                    generation=strategy.generation + 1,
                    status=StrategyStatus.CREATED,
                )
                await db.save_strategy(repaired_strategy)
                await db.update_strategy_status(repaired_strategy.id, StrategyStatus.BACKTESTING)

                retry_run = await runner.run(
                    repaired_strategy,
                    pair,
                    timeframe,
                    settings.in_sample_start,
                    settings.in_sample_end,
                )
                await db.save_backtest_run(retry_run)

                await db.update_strategy_status(strategy.id, StrategyStatus.ARCHIVED)
                strategy = repaired_strategy
                run = retry_run
                log.info(
                    "strategy.runtime_retry.completed",
                    original_strategy=strategy.parent_ids[0] if strategy.parent_ids else None,
                    repaired_strategy=strategy.name,
                    disqualified=run.disqualified,
                    reason=run.disqualification_reason,
                )
            except Exception as repair_error:
                log.warning(
                    "strategy.runtime_retry.failed",
                    strategy=strategy.name,
                    error=str(repair_error),
                )

        if run.disqualified:
            await db.update_strategy_status(strategy.id, StrategyStatus.FAILED)
            # Gap 1: analyze failure structure before closing
            await failure_analyzer.analyze(
                strategy_code=strategy.code,
                metrics=_metrics_to_dict(run.metrics),
                disqualification_reason=run.disqualification_reason,
                outcome_summary=f"Disqualified: {run.disqualification_reason}",
                context=context,
                hypothesis_type=hypothesis.type.value,
                run_ids=[run.id],
            )
            return (
                HypothesisOutcome.REFUTED,
                f"Disqualified: {run.disqualification_reason}",
                strategy.id,
            )

        # --- Walk-forward gate (if enabled) ---
        if settings.walk_forward_enabled:
            wf_passed, wf_summary = await self._run_walk_forward(strategy, pair, timeframe)
            if not wf_passed:
                await db.update_strategy_status(strategy.id, StrategyStatus.ARCHIVED)
                await failure_analyzer.analyze(
                    strategy_code=strategy.code,
                    metrics=_metrics_to_dict(run.metrics),
                    disqualification_reason=f"Walk-forward failed: {wf_summary}",
                    outcome_summary=wf_summary,
                    context=context,
                    hypothesis_type=hypothesis.type.value,
                    run_ids=[run.id],
                )
                return (
                    HypothesisOutcome.REFUTED,
                    f"Failed walk-forward: {wf_summary}",
                    strategy.id,
                )
            log.info("agent.walk_forward_passed", strategy=strategy.name, summary=wf_summary)

        # --- Holdout run (sacred — only reached after walk-forward passes) ---
        holdout = await runner.run(
            strategy, pair, timeframe,
            settings.holdout_start, settings.holdout_end,
            is_holdout=True,
        )
        if holdout.metrics.sharpe is not None and run.metrics.sharpe is not None:
            holdout.oos_sharpe_delta = run.metrics.sharpe - holdout.metrics.sharpe
        await db.save_backtest_run(holdout)

        await db.update_strategy_status(
            strategy.id, StrategyStatus.EVALUATED, best_score=run.composite_score
        )
        await self._credit_components(strategy, pair, timeframe, run)

        score = run.composite_score or 0
        oos_note = ""
        if holdout.oos_sharpe_delta is not None:
            oos_note = f" | OOS delta: {holdout.oos_sharpe_delta:+.2f}"

        outcome = (
            HypothesisOutcome.CONFIRMED if score > 0.5
            else HypothesisOutcome.INCONCLUSIVE if score > 0
            else HypothesisOutcome.REFUTED
        )
        summary = (
            f"Score: {score:.3f} | Sharpe: {run.metrics.sharpe:.2f} | "
            f"DD: {run.metrics.max_drawdown:.1%} | Trades: {run.metrics.trade_count}"
            f"{oos_note}"
        )

        # Gap 1: analyze failures even when not hard-disqualified but scored poorly
        if outcome == HypothesisOutcome.REFUTED:
            await failure_analyzer.analyze(
                strategy_code=strategy.code,
                metrics=_metrics_to_dict(run.metrics),
                disqualification_reason=None,
                outcome_summary=summary,
                context=context,
                hypothesis_type=hypothesis.type.value,
                run_ids=[run.id, holdout.id],
            )

        return outcome, summary, strategy.id

    def _is_runtime_disqualification(self, reason: str | None) -> bool:
        if not reason:
            return False
        lowered = reason.lower()
        return (
            "freqtrade exited with code" in lowered
            or "traceback" in lowered
            or "typeerror" in lowered
            or "syntaxerror" in lowered
        )

    async def _run_exploit(self, hypothesis):
        """Bayesian parameter optimization using Optuna — enriched with failure context."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        strategy_row = await db.get_strategy(hypothesis.target_strategy_id)
        if not strategy_row:
            return HypothesisOutcome.INCONCLUSIVE, "Target strategy not found", None

        context = hypothesis.context
        pair = context.get("pair", settings.backtest_pairs[0])
        timeframe = context.get("timeframe", settings.backtest_timeframes[0])
        best_score = [strategy_row.get("best_score") or 0]
        best_strategy_id = [hypothesis.target_strategy_id]

        # Gap 2: fetch recent failures + directives for this context upfront
        recent_failures = await db.get_recent_failures_for_context(pair, timeframe, limit=3)
        failure_directives = await db.get_failure_directives(pair, timeframe, limit=10)
        trial_scores: list[dict] = []  # track what we've tried

        async def objective_wrapper(trial: optuna.Trial) -> float:
            # Gap 2: inform LLM of the failure landscape before suggesting values
            failures_text = _format_recent_failures(recent_failures)
            directives_text = _format_directives(failure_directives)
            tried_text = _format_tried(trial_scores)

            param_prompt = f"""
Optuna trial #{trial.number} — suggest parameter values for strategy optimization.

Strategy: {strategy_row['name']}
Current best score: {best_score[0]:.3f}
Pair: {pair} | Timeframe: {timeframe}
Parameter hints: {hypothesis.parameter_changes}

=== RECENT FAILURES ON THIS CONTEXT (avoid repeating these) ===
{failures_text}

=== ACTIVE FAILURE CONSTRAINTS ===
{directives_text}

=== PARAMETER VALUES ALREADY TRIED THIS SESSION ===
{tried_text}

Based on the above, suggest parameter values that:
1. Avoid the failure zones documented above
2. Are meaningfully different from already-tried values
3. Address the root causes of recent failures (e.g. if trade count is too low, loosen entry thresholds)

Return JSON: {{"param_name": value, ...}}
"""
            from src.llm.client import llm
            suggested = await llm.complete_json(
                system="You are a trading strategy parameter optimizer.",
                user=param_prompt,
                temperature=0.3,
            )

            from src.models import Strategy, StrategyStatus
            tuned_code = await self._apply_suggested_parameters(
                strategy_row["code"], suggested
            )
            new_strategy = Strategy(
                id=uuid.uuid4(),
                name=f"{strategy_row['name']}_opt_{trial.number}",
                code=tuned_code,
                component_ids=[
                    uuid.UUID(str(c))
                    for c in (strategy_row.get("component_ids") or [])
                ],
                parameters={**dict(strategy_row.get("parameters") or {}), **suggested},
                hypothesis_id=hypothesis.id,
                parent_ids=[hypothesis.target_strategy_id],
                generation=(strategy_row.get("generation") or 0) + 1,
                status=StrategyStatus.CREATED,
            )
            await db.save_strategy(new_strategy)

            run = await runner.run(
                new_strategy, pair, timeframe,
                settings.in_sample_start, settings.in_sample_end,
            )
            await db.save_backtest_run(run)

            score = run.composite_score or -1.0
            trial_scores.append({
                "trial": trial.number,
                "params": suggested,
                "score": score,
                "disqualified": run.disqualified,
                "reason": run.disqualification_reason,
            })

            if score > best_score[0]:
                best_score[0] = score
                best_strategy_id[0] = new_strategy.id
                await db.update_strategy_status(
                    new_strategy.id, StrategyStatus.EVALUATED, best_score=score
                )
            elif run.disqualified:
                # Analyze each failed trial to accumulate per-trial failure patterns
                await failure_analyzer.analyze(
                    strategy_code=new_strategy.code,
                    metrics=_metrics_to_dict(run.metrics),
                    disqualification_reason=run.disqualification_reason,
                    outcome_summary=f"Optuna trial {trial.number} disqualified",
                    context=context,
                    hypothesis_type=hypothesis.type.value,
                    run_ids=[run.id],
                )
            return score

        study = optuna.create_study(direction="maximize")
        for i in range(5):
            trial = study.ask()
            score = await objective_wrapper(trial)
            study.tell(trial, score)

        original_score = strategy_row.get("best_score") or 0
        outcome = (
            HypothesisOutcome.CONFIRMED if best_score[0] > original_score
            else HypothesisOutcome.INCONCLUSIVE
        )
        summary = (
            f"Best optimized score: {best_score[0]:.3f} after 5 trials "
            f"(was {original_score:.3f})"
        )
        return outcome, summary, best_strategy_id[0]

    async def _run_ablate(self, hypothesis):
        """Remove a component from a strategy and measure the delta."""
        strategy_row = await db.get_strategy(hypothesis.target_strategy_id)
        if not strategy_row:
            return HypothesisOutcome.INCONCLUSIVE, "Target strategy not found", None

        context = hypothesis.context
        pair = context.get("pair", "BTC/USDT")
        timeframe = context.get("timeframe", "4h")

        # Get original run score / Sharpe baseline
        runs = await db.get_runs_for_strategy(hypothesis.target_strategy_id)
        original_score = max((r.get("composite_score") or 0 for r in runs), default=0)
        original_sharpe = max((r.get("sharpe") or 0 for r in runs), default=0)

        # Ask LLM to produce a version of the strategy without the target component
        target_component_ids = hypothesis.target_component_ids
        components = await self._load_components(target_component_ids)
        component_names = [c.name for c in components] if components else []

        ablation_prompt = f"""
Remove the following component(s) from this Freqtrade strategy and return the modified code.

Components to remove: {component_names}
Original strategy code:
{strategy_row['code']}

Return ONLY the modified Python code, no explanations.
"""
        ablated_code = await llm.complete(
            system="You are a Freqtrade strategy developer. Modify the strategy as instructed.",
            user=ablation_prompt,
            temperature=0.05,
        )

        from src.models import Strategy, StrategyStatus
        ablated_strategy = Strategy(
            id=uuid.uuid4(),
            name=f"{strategy_row['name']}_ablated",
            code=ablated_code.strip(),
            component_ids=[],
            parameters=dict(strategy_row.get("parameters") or {}),
            hypothesis_id=hypothesis.id,
            parent_ids=[hypothesis.target_strategy_id],
            generation=(strategy_row.get("generation") or 0) + 1,
            status=StrategyStatus.CREATED,
        )
        await db.save_strategy(ablated_strategy)

        run = await runner.run(
            ablated_strategy, pair, timeframe, settings.in_sample_start, settings.in_sample_end
        )
        await db.save_backtest_run(run)

        ablated_score = run.composite_score or 0
        sharpe_delta = (run.metrics.sharpe or 0) - (original_sharpe or 0)

        # Update component performance with ablation result
        for comp in components:
            cp = ComponentPerformance(
                id=uuid.uuid4(),
                component_id=comp.id,
                pair=pair,
                timeframe=timeframe,
                regime_trend=RegimeTrend.UNKNOWN,
                regime_vol=RegimeVol.UNKNOWN,
                appearances=0,
                ablation_count=1,
                avg_sharpe_contribution=sharpe_delta,
                confidence=0.5,
            )
            await db.update_component_performance(cp)

        summary = (
            f"Without {component_names}: score={ablated_score:.3f} "
            f"(was {original_score:.3f}), sharpe_delta={sharpe_delta:+.3f}"
        )
        outcome = HypothesisOutcome.CONFIRMED if abs(sharpe_delta) > 0.1 else HypothesisOutcome.INCONCLUSIVE
        return outcome, summary, ablated_strategy.id

    async def _run_crossover(self, hypothesis):
        """
        Semantic component-level crossover.

        Gap 3: secondary parent is fitness-weighted selected from the population.
        Gap 4: crossover happens at the component level, not raw code truncation.
        The LLM assembles explicit component lists from each parent — not free-form merging.
        """
        import random

        # --- Primary parent: from hypothesis ---
        strategy_row = await db.get_strategy(hypothesis.target_strategy_id)
        if not strategy_row:
            return HypothesisOutcome.INCONCLUSIVE, "Primary parent not found", None

        # --- Gap 3: fitness-proportionate selection of secondary parent ---
        population = await db.get_strategies_fitness_weighted(limit=20)
        # Exclude the primary parent
        candidates = [
            s for s in population
            if str(s["strategy_id"]) != str(hypothesis.target_strategy_id)
        ]
        if not candidates:
            return HypothesisOutcome.INCONCLUSIVE, "No secondary parent candidates", None

        secondary_row = _fitness_tournament_select(candidates, tournament_size=3)

        # --- Gap 4: Component-level crossover ---
        # Resolve component lists for each parent
        all_components = await db.get_all_components()
        comp_map = {str(c["id"]): c for c in all_components}

        parent_a_comp_ids = [str(c) for c in (strategy_row.get("component_ids") or [])]
        parent_b_comp_ids = [str(c) for c in (secondary_row.get("component_ids") or [])]

        parent_a_components = [comp_map[cid] for cid in parent_a_comp_ids if cid in comp_map]
        parent_b_components = [comp_map[cid] for cid in parent_b_comp_ids if cid in comp_map]

        # If we have component metadata, do semantic crossover
        # Otherwise fall back to code-level merge (but still context-aware)
        insights = await db.get_active_insights()
        failure_directives = await db.get_failure_directives(limit=10)

        if parent_a_components and parent_b_components:
            # True semantic crossover: ask LLM to assign components to child
            crossover_plan_prompt = f"""
Design a component-level crossover between two Freqtrade strategies.

Parent A: {strategy_row['name']} (score: {strategy_row.get('best_score', 0):.3f})
Components: {_format_component_list(parent_a_components)}

Parent B: {secondary_row['name']} (score: {secondary_row.get('composite_score', 0):.3f})
Components: {_format_component_list(parent_b_components)}

Active failure constraints (MUST AVOID):
{_format_directives(failure_directives)}

Active insights (follow these):
{chr(10).join(f"- {i['statement']}" for i in insights[:8])}

Task: Select which components from each parent to include in the child strategy.
Rules:
- Child must have at most {settings.max_components_per_strategy} components total
- Must include exactly one entry_signal, one exit_signal
- Can mix indicator/filter/risk_rule from either parent
- Avoid component combinations flagged in failure constraints
- Prefer components with higher individual performance scores

Return JSON:
{{
  "from_parent_a": ["component_name_1", ...],
  "from_parent_b": ["component_name_2", ...],
  "rationale": "Why this combination should work",
  "expected_improvement": "What specific weakness of each parent this addresses"
}}
"""
            from src.llm.client import llm
            plan = await llm.complete_json(
                system="You are a quantitative trading strategy researcher.",
                user=crossover_plan_prompt,
                temperature=0.15,
            )

            from_a_names = set(plan.get("from_parent_a", []))
            from_b_names = set(plan.get("from_parent_b", []))

            child_components = (
                [c for c in parent_a_components if c["name"] in from_a_names]
                + [c for c in parent_b_components if c["name"] in from_b_names]
            )
            child_component_ids = [uuid.UUID(str(c["id"])) for c in child_components]
            child_components = await self._load_components(child_component_ids)
            crossover_rationale = plan.get("rationale", "Component-level crossover")

        else:
            # No component metadata — fall back but at least document the merge
            child_components = []
            child_component_ids = []
            crossover_rationale = hypothesis.parameter_changes.get("merge_approach", "Code-level merge")

        # Generate the child strategy code
        from src.llm.client import llm
        if child_components:
            # Component-first generation — assemble from explicit parts
            child_strategy = await generator.generate_from_hypothesis(
                hypothesis=hypothesis,
                components=child_components,
                existing_insights=insights,
            )
        else:
            # Code-level fallback
            merge_prompt = f"""
Merge these two Freqtrade strategies. Take entry logic from A, exit + risk from B.
Follow these constraints: {_format_directives(failure_directives)}

Strategy A:
{strategy_row['code'][:2500]}

Strategy B:
{secondary_row['code'][:2500]}

Return ONLY the complete merged Python code.
Name the class: Crossover_{uuid.uuid4().hex[:6]}
"""
            merged_code = await llm.complete(
                system="You are an expert Freqtrade strategy developer.",
                user=merge_prompt,
                temperature=0.15,
                max_tokens=6000,
            )
            from src.models import Strategy, StrategyStatus
            child_strategy = Strategy(
                id=uuid.uuid4(),
                name=f"Crossover_{uuid.uuid4().hex[:8]}",
                code=merged_code.strip(),
                component_ids=[],
                parameters={},
                hypothesis_id=hypothesis.id,
                parent_ids=[hypothesis.target_strategy_id,
                            uuid.UUID(str(secondary_row["strategy_id"]))],
                generation=max(
                    strategy_row.get("generation") or 0,
                    secondary_row.get("generation") or 0,
                ) + 1,
                status=StrategyStatus.CREATED,
            )

        if child_component_ids:
            child_strategy.component_ids = child_component_ids
            child_strategy.parent_ids = [
                hypothesis.target_strategy_id,
                uuid.UUID(str(secondary_row["strategy_id"])),
            ]
            child_strategy.generation = max(
                strategy_row.get("generation") or 0,
                secondary_row.get("generation") or 0,
            ) + 1

        await db.save_strategy(child_strategy)

        context = hypothesis.context
        run = await runner.run(
            child_strategy,
            context.get("pair", settings.backtest_pairs[0]),
            context.get("timeframe", settings.backtest_timeframes[0]),
            settings.in_sample_start,
            settings.in_sample_end,
        )
        await db.save_backtest_run(run)

        parent_a_score = strategy_row.get("best_score") or 0
        parent_b_score = secondary_row.get("composite_score") or 0
        min_parent_score = min(parent_a_score, parent_b_score)
        score = run.composite_score or 0

        if score <= min_parent_score:
            await failure_analyzer.analyze(
                strategy_code=child_strategy.code,
                metrics=_metrics_to_dict(run.metrics),
                disqualification_reason=run.disqualification_reason,
                outcome_summary=f"Crossover underperformed parents: {score:.3f} vs {min_parent_score:.3f}",
                context=context,
                hypothesis_type=hypothesis.type.value,
                run_ids=[run.id],
            )

        outcome = (
            HypothesisOutcome.CONFIRMED if score > min_parent_score
            else HypothesisOutcome.REFUTED
        )
        n_child_comps = len(child_component_ids) or "?"
        summary = (
            f"Crossover score: {score:.3f} vs parents "
            f"{parent_a_score:.3f}/{parent_b_score:.3f} | "
            f"child components: {n_child_comps} | {crossover_rationale[:60]}"
        )
        return outcome, summary, child_strategy.id

    async def _run_stress_test(self, hypothesis):
        """Run strategy across multiple contexts."""
        strategy_row = await db.get_strategy(hypothesis.target_strategy_id)
        if not strategy_row:
            return HypothesisOutcome.INCONCLUSIVE, "Target strategy not found", None

        from src.models import Strategy, StrategyStatus
        strategy = Strategy(
            id=uuid.UUID(str(strategy_row["id"])),
            name=strategy_row["name"],
            code=strategy_row["code"],
            component_ids=[],
            parameters=dict(strategy_row.get("parameters") or {}),
            status=StrategyStatus.EVALUATED,
        )

        contexts = hypothesis.parameter_changes.get("additional_contexts", [])
        results = []
        for ctx in contexts:
            run = await runner.run(
                strategy,
                ctx.get("pair", "BTC/USDT"),
                ctx.get("timeframe", "4h"),
                settings.in_sample_start,
                settings.in_sample_end,
            )
            await db.save_backtest_run(run)
            results.append(run)

        passed = sum(1 for r in results if not r.disqualified and (r.composite_score or 0) > 0)
        robustness = passed / max(len(results), 1)
        outcome = (
            HypothesisOutcome.CONFIRMED if robustness >= 0.6
            else HypothesisOutcome.REFUTED if robustness < 0.3
            else HypothesisOutcome.INCONCLUSIVE
        )
        summary = f"Robustness: {robustness:.0%} ({passed}/{len(results)} contexts profitable)"
        return outcome, summary, hypothesis.target_strategy_id

    async def _run_regime_specialize(self, hypothesis):
        """Add regime filter to a strategy."""
        strategy_row = await db.get_strategy(hypothesis.target_strategy_id)
        if not strategy_row:
            return HypothesisOutcome.INCONCLUSIVE, "Target strategy not found", None

        regime_type = hypothesis.parameter_changes.get("regime_type", "trend")
        threshold = hypothesis.parameter_changes.get("filter_threshold", 25)

        regime_prompt = f"""
Add a regime filter to this Freqtrade strategy.

Regime filter type: {regime_type}
Threshold: {threshold}

For a trend regime filter, add ADX > {threshold} as an entry condition.
For a volatility regime filter, add appropriate Bollinger Band width or ATR conditions.

Original strategy:
{strategy_row['code']}

Return ONLY the modified Python code.
"""
        modified_code = await llm.complete(
            system="You are an expert Freqtrade strategy developer.",
            user=regime_prompt,
            temperature=0.05,
            max_tokens=6000,
        )

        from src.models import Strategy, StrategyStatus
        specialized = Strategy(
            id=uuid.uuid4(),
            name=f"{strategy_row['name']}_regime_{regime_type}",
            code=modified_code.strip(),
            component_ids=[],
            parameters=dict(strategy_row.get("parameters") or {}),
            hypothesis_id=hypothesis.id,
            parent_ids=[hypothesis.target_strategy_id],
            generation=(strategy_row.get("generation") or 0) + 1,
            status=StrategyStatus.CREATED,
        )
        await db.save_strategy(specialized)

        context = hypothesis.context
        run = await runner.run(
            specialized,
            context.get("pair", "BTC/USDT"),
            context.get("timeframe", "4h"),
            settings.in_sample_start,
            settings.in_sample_end,
        )
        await db.save_backtest_run(run)

        original_score = strategy_row.get("best_score") or 0
        new_score = run.composite_score or 0
        outcome = (
            HypothesisOutcome.CONFIRMED if new_score > original_score
            else HypothesisOutcome.INCONCLUSIVE
        )
        summary = f"Regime specialized score: {new_score:.3f} vs original {original_score:.3f}"
        return outcome, summary, specialized.id

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    async def _run_walk_forward(
        self, strategy, pair: str, timeframe: str
    ) -> tuple[bool, str]:
        """
        Run the strategy across all walk-forward windows.
        Returns (passed, summary_string).
        A strategy passes if >= walk_forward_min_pass_ratio windows are profitable
        and not disqualified.
        """
        windows = settings.walk_forward_windows
        if not windows:
            return True, "No walk-forward windows configured"

        results = []
        for wf_start, wf_end in windows:
            wf_run = await runner.run(
                strategy, pair, timeframe, wf_start, wf_end, is_holdout=False
            )
            # Tag as a walk-forward run in raw_output for later querying
            wf_run.raw_output["walk_forward"] = True
            wf_run.raw_output["wf_window"] = f"{wf_start}:{wf_end}"
            await db.save_backtest_run(wf_run)
            results.append(wf_run)

        n_total = len(results)
        n_passed = sum(
            1 for r in results
            if not r.disqualified and (r.composite_score or 0) > 0
        )
        pass_ratio = n_passed / n_total if n_total > 0 else 0
        passed = pass_ratio >= settings.walk_forward_min_pass_ratio

        sharpes = [r.metrics.sharpe for r in results if r.metrics.sharpe is not None]
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0

        summary = (
            f"{n_passed}/{n_total} windows profitable "
            f"({pass_ratio:.0%} vs threshold {settings.walk_forward_min_pass_ratio:.0%}) "
            f"| avg window Sharpe: {avg_sharpe:.2f}"
        )
        return passed, summary

    async def _apply_suggested_parameters(
        self, strategy_code: str, suggested: dict
    ) -> str:
        """
        Render a strategy variant with concrete parameter defaults for this trial.
        If rendering fails, return the original code so the trial can still run.
        """
        if not suggested:
            return strategy_code

        prompt = f"""
You are editing a Freqtrade strategy class.

Task:
- Apply these parameter values to the strategy code as concrete defaults:
{suggested}
- Preserve strategy behavior and structure except for parameter value updates.
- Keep valid runnable Python.
- Return only the full updated Python code.

Original strategy code:
{strategy_code}
"""
        try:
            tuned = await llm.complete(
                system="You are an expert Freqtrade strategy developer.",
                user=prompt,
                temperature=0.0,
                max_tokens=6000,
            )
            tuned = tuned.strip()
            if tuned.startswith("```"):
                tuned = "\n".join(tuned.split("\n")[1:])
            if tuned.endswith("```"):
                tuned = "\n".join(tuned.split("\n")[:-1])
            return tuned.strip() or strategy_code
        except Exception as e:
            log.warning("agent.exploit_param_render_failed", error=str(e))
            return strategy_code

    async def _load_components(self, component_ids: list) -> list[Component]:
        all_components = await db.get_all_components()
        if component_ids:
            id_strs = [str(c) for c in component_ids]
            rows = [c for c in all_components if str(c["id"]) in id_strs]
        else:
            rows = all_components

        components: list[Component] = []
        for row in rows:
            components.append(
                Component(
                    id=uuid.UUID(str(row["id"])),
                    category=ComponentCategory(row["category"]),
                    name=row["name"],
                    code_snippet=row["code_snippet"],
                    parameters=dict(row.get("parameters") or {}),
                    parameter_space=dict(row.get("parameter_space") or {}),
                    dependencies=list(row.get("dependencies") or []),
                    description=row.get("description") or "",
                    origin=ComponentOrigin(row["origin"]),
                    parent_id=(
                        uuid.UUID(str(row["parent_id"]))
                        if row.get("parent_id") else None
                    ),
                    tags=list(row.get("tags") or []),
                    created_at=row.get("created_at") or datetime.now(timezone.utc),
                )
            )
        return components

    async def _credit_components(self, strategy, pair, timeframe, run):
        """Increment appearance count for all components in a strategy."""
        if not strategy.component_ids:
            return
        all_comps = await db.get_all_components()
        comp_map = {str(c["id"]): c for c in all_comps}
        for cid in strategy.component_ids:
            comp = comp_map.get(str(cid))
            if not comp:
                continue
            cp = ComponentPerformance(
                id=uuid.uuid4(),
                component_id=uuid.UUID(str(cid)),
                pair=pair,
                timeframe=timeframe,
                regime_trend=run.regime_trend,
                regime_vol=run.regime_vol,
                appearances=1,
                ablation_count=0,
                confidence=0.1,
            )
            await db.update_component_performance(cp)

    def _row_to_hypothesis(self, row: dict):
        from src.models import Hypothesis, HypothesisType, HypothesisStatus
        return Hypothesis(
            id=uuid.UUID(str(row["id"])),
            type=HypothesisType(row["type"]),
            rationale=row["rationale"],
            prediction=row.get("prediction") or "",
            priority_score=row.get("priority_score") or 0,
            target_component_ids=[
                uuid.UUID(str(c)) for c in (row.get("target_component_ids") or [])
            ],
            target_strategy_id=(
                uuid.UUID(str(row["target_strategy_id"]))
                if row.get("target_strategy_id") else None
            ),
            parameter_changes=row.get("parameter_changes") or {},
            context=row.get("context") or {},
            status=HypothesisStatus(row["status"]),
        )


# ============================================================
# Module-level helpers
# ============================================================

def _metrics_to_dict(metrics) -> dict:
    """Convert a BacktestMetrics dataclass to a plain dict for the failure analyzer."""
    return {
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "profit_factor": metrics.profit_factor,
        "max_drawdown": metrics.max_drawdown,
        "win_rate": metrics.win_rate,
        "trade_count": metrics.trade_count,
        "avg_trade_dur_h": metrics.avg_trade_dur_h,
        "calmar": metrics.calmar,
        "oos_sharpe_delta": None,  # set by caller when available
        "trade_count_stability": None,
    }


def _fitness_tournament_select(
    candidates: list[dict], tournament_size: int = 3
) -> dict:
    """
    Gap 3: Tournament selection — pick tournament_size random candidates,
    return the one with the highest composite_score.
    Gives high-fitness strategies more selection probability without being
    purely greedy (which would collapse diversity).
    """
    import random
    tournament = random.sample(candidates, min(tournament_size, len(candidates)))
    return max(tournament, key=lambda s: s.get("composite_score") or 0)


def _format_recent_failures(failures: list[dict]) -> str:
    if not failures:
        return "  (no prior failures on this context)"
    lines = []
    for f in failures:
        lines.append(
            f"  - {f['strategy_name']}: sharpe={f['sharpe']}, "
            f"dd={f['max_drawdown']}, trades={f['trade_count']}, "
            f"reason={f.get('disqualification_reason', 'poor score')} "
            f"| params={f.get('parameters', {})}"
        )
    return "\n".join(lines)


def _format_directives(directives: list[dict]) -> str:
    if not directives:
        return "  (no active constraints)"
    return "\n".join([
        f"  [{d['root_cause']}] {d['constraint_text']} (confidence: {d['confidence']:.2f})"
        for d in directives
    ])


def _format_tried(trial_scores: list[dict]) -> str:
    if not trial_scores:
        return "  (first trial)"
    lines = []
    for t in trial_scores:
        status = "DISQ" if t.get("disqualified") else f"score={t['score']:.3f}"
        lines.append(f"  trial {t['trial']}: {t['params']} → {status}")
    return "\n".join(lines)


def _format_component_list(components: list[dict]) -> str:
    formatted = []
    for c in components:
        if isinstance(c, dict):
            name = c.get("name", "unknown")
            category = c.get("category", "unknown")
        else:
            name = getattr(c, "name", "unknown")
            category_obj = getattr(c, "category", "unknown")
            category = getattr(category_obj, "value", category_obj)
        formatted.append(f"{name} [{category}]")
    return ", ".join(formatted) or "(none)"


# ============================================================
# CLI entrypoint
# ============================================================

async def main():
    import logging
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        )
    )

    parser = argparse.ArgumentParser(description="Freqtrade Strategy Research Agent")
    parser.add_argument("--mode", choices=["run", "loop", "synthesize"], default="run")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument(
        "--hypothesis-type",
        choices=[t.value for t in HypothesisType],
        default=None,
    )
    args = parser.parse_args()

    await db.connect()
    agent = AgentLoop()

    try:
        if args.mode == "run":
            await agent.startup_checks()
            await agent.run_once(force_type=args.hypothesis_type)
        elif args.mode == "loop":
            await agent.run_loop(max_experiments=args.max_experiments)
        elif args.mode == "synthesize":
            await synthesizer.synthesize()
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

"""
Hypothesis Engine — the strategic brain of the agent.

Decides WHAT to test next and WHY, by reading the knowledge base and
computing a priority-ranked queue of experiments.

Supports 6 experiment modes:
  EXPLORE         — novel component combinations never tested together
  EXPLOIT         — Bayesian parameter optimization on a known-good strategy
  ABLATE          — isolate a component's marginal contribution
  CROSSOVER       — merge elements from two parent strategies
  STRESS_TEST     — run a promising strategy across more contexts
  REGIME_SPECIALIZE — add regime filter to a high-variance strategy
"""
from __future__ import annotations
import math
import uuid
from datetime import datetime
from typing import Any, Optional

import structlog

from src.config import settings
from src.db.database import db
from src.llm.client import llm
from src.models import (
    Hypothesis, HypothesisOutcome, HypothesisStatus, HypothesisType,
)

log = structlog.get_logger()


class HypothesisEngine:
    """
    Generates, prioritizes, and closes hypotheses.
    The engine never runs experiments itself — it produces Hypothesis objects
    that the agent loop dispatches to the backtest pipeline.
    """

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def generate_next(self, force_type: str = None) -> Optional[Hypothesis]:
        """
        Decide what to test next. Returns the highest-priority new Hypothesis,
        or None if nothing useful can be generated right now.
        """
        kb = await db.get_knowledge_summary()
        candidates: list[Hypothesis] = []

        if force_type:
            types = [HypothesisType(force_type)]
        else:
            types = await self._select_experiment_types(kb)

        for h_type in types:
            h = await self._generate_hypothesis(h_type, kb)
            if h:
                candidates.append(h)

        if not candidates:
            log.warning("hypothesis.no_candidates")
            return None

        best = max(candidates, key=lambda h: h.priority_score)
        await db.save_hypothesis(best)
        log.info(
            "hypothesis.generated",
            type=best.type.value,
            priority=best.priority_score,
            rationale=best.rationale[:80],
        )
        return best

    async def close(
        self,
        hypothesis: Hypothesis,
        outcome: HypothesisOutcome,
        summary: str,
        resulting_strategy_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Mark a hypothesis as complete and record the outcome."""
        hypothesis.status = HypothesisStatus.COMPLETE
        hypothesis.outcome = outcome
        hypothesis.outcome_summary = summary
        hypothesis.resulting_strategy_id = resulting_strategy_id
        hypothesis.completed_at = datetime.utcnow()
        await db.update_hypothesis(hypothesis)
        log.info(
            "hypothesis.closed",
            id=str(hypothesis.id),
            outcome=outcome.value,
            summary=summary[:80],
        )

    # --------------------------------------------------------
    # Experiment type selection
    # --------------------------------------------------------

    async def _select_experiment_types(
        self, kb: dict[str, Any]
    ) -> list[HypothesisType]:
        """
        Choose which experiment types to generate candidates for.

        Gap 5: Base weights are now overridden by engine_directives produced
        by insight synthesis — the agent's own analysis of what to do next
        directly drives sampling, not just static recency heuristics.
        """
        total = kb["total_runs"]

        # Bootstrap phase: always explore first
        if total < 20:
            return [HypothesisType.EXPLORE]

        # Count how many of each type ran recently
        recent_counts = {}
        for h_type in HypothesisType:
            recent_counts[h_type] = await db.count_recent_by_type(h_type, n=10)

        # Base weights — diversity heuristic baseline
        weights = {
            HypothesisType.EXPLORE: max(0.30 - recent_counts[HypothesisType.EXPLORE] * 0.05, 0.05),
            HypothesisType.EXPLOIT: max(0.30 - recent_counts[HypothesisType.EXPLOIT] * 0.05, 0.05),
            HypothesisType.ABLATE: max(0.15 - recent_counts[HypothesisType.ABLATE] * 0.03, 0.02),
            HypothesisType.CROSSOVER: max(0.10 - recent_counts[HypothesisType.CROSSOVER] * 0.02, 0.01),
            HypothesisType.STRESS_TEST: max(0.10 - recent_counts[HypothesisType.STRESS_TEST] * 0.02, 0.01),
            HypothesisType.REGIME_SPECIALIZE: 0.05,
        }

        # Gap 5: apply engine directives produced by insight synthesis
        directives = await db.get_engine_directives()
        for d in directives:
            try:
                h_type = HypothesisType(d["target_hypothesis_type"])
                bias = float(d.get("bias_weight", 0.2))
                if d["directive_type"] == "boost":
                    weights[h_type] = min(weights[h_type] + bias, 1.0)
                    log.debug(
                        "engine.directive_applied",
                        type="boost",
                        target=h_type.value,
                        delta=bias,
                    )
                elif d["directive_type"] == "suppress":
                    weights[h_type] = max(weights[h_type] - bias, 0.01)
                    log.debug(
                        "engine.directive_applied",
                        type="suppress",
                        target=h_type.value,
                        delta=bias,
                    )
            except (ValueError, KeyError):
                continue

        # Pick top 3 by adjusted weight
        sorted_types = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in sorted_types[:3]]

        log.info(
            "engine.type_selection",
            selected=[t.value for t in selected],
            weights={t.value: round(w, 3) for t, w in sorted_types},
            active_directives=len(directives),
        )
        return selected

    # --------------------------------------------------------
    # Hypothesis generators (one per type)
    # --------------------------------------------------------

    async def _generate_hypothesis(
        self, h_type: HypothesisType, kb: dict[str, Any]
    ) -> Optional[Hypothesis]:
        generators = {
            HypothesisType.EXPLORE: self._gen_explore,
            HypothesisType.EXPLOIT: self._gen_exploit,
            HypothesisType.ABLATE: self._gen_ablate,
            HypothesisType.CROSSOVER: self._gen_crossover,
            HypothesisType.STRESS_TEST: self._gen_stress_test,
            HypothesisType.REGIME_SPECIALIZE: self._gen_regime_specialize,
        }
        try:
            return await generators[h_type](kb)
        except Exception as e:
            log.error("hypothesis.generation_error", type=h_type.value, error=str(e))
            return None

    async def _gen_explore(self, kb: dict[str, Any]) -> Optional[Hypothesis]:
        """Generate a hypothesis to explore a novel component combination."""
        components = await db.get_all_components()
        if len(components) < 2:
            return self._bootstrap_explore(kb)

        scores = await db.get_component_scores()
        score_map = {s["name"]: s["avg_sharpe_contrib"] for s in scores if s["avg_sharpe_contrib"]}
        available_contexts = settings.available_contexts
        coverage_map = {(c["pair"], c["timeframe"]): c["runs"] for c in kb["coverage"]}
        failure_directives = await db.get_failure_directives(limit=15)

        prompt = f"""
You are designing a trading strategy experiment.

Available components and their performance scores:
{_format_components(components, score_map)}

Available test contexts (pair + timeframe combinations you may choose from):
{_format_contexts(available_contexts, coverage_map)}

Knowledge base summary:
- Total runs so far: {kb['total_runs']}
- Active insights: {_format_insights(kb['active_insights'])}

=== ACTIVE FAILURE CONSTRAINTS (these combinations have been proven to fail) ===
{chr(10).join(f"  [{d['root_cause']}] {d['constraint_text']}" for d in failure_directives) or "  (none yet)"}

Task:
1. Select 2-4 components that have not been combined before and would form coherent logic.
   Prefer components with positive scores. Prefer complementary roles (e.g. indicator + entry + filter).
   AVOID component combinations that match any failure constraint above.
2. Select the best context to test in. Prefer under-explored pairs/timeframes (low run count).
   You MUST choose from the available_contexts list above only.

Return JSON:
{{
  "selected_component_names": ["name1", "name2", ...],
  "rationale": "Why these components are worth combining",
  "prediction": "Expected direction of Sharpe/drawdown change",
  "target_context": {{"pair": "BTC/USDT", "timeframe": "4h"}},
  "context_rationale": "Why this pair/timeframe was chosen"
}}
"""
        result = await llm.complete_json(
            system="You are a quantitative trading strategy researcher.",
            user=prompt,
        )

        selected_ids = [
            c["id"] for c in components
            if c["name"] in result.get("selected_component_names", [])
        ]

        # Validate the LLM chose a real context; fall back to least-covered if not
        chosen = result.get("target_context", {})
        context = self._validate_context(chosen, available_contexts, coverage_map)
        priority = self._priority_explore(kb, score_map, result.get("selected_component_names", []))

        return Hypothesis(
            id=uuid.uuid4(),
            type=HypothesisType.EXPLORE,
            rationale=result.get("rationale", "Exploring novel component combination"),
            prediction=result.get("prediction", ""),
            priority_score=priority,
            target_component_ids=selected_ids,
            target_strategy_id=None,
            parameter_changes={},
            context=context,
            status=HypothesisStatus.QUEUED,
        )

    async def _gen_exploit(self, kb: dict[str, Any]) -> Optional[Hypothesis]:
        """Tune parameters on a known-good strategy."""
        top = await db.get_top_strategies(limit=5)
        if not top:
            return None

        # Pick the top strategy that hasn't been exploited recently
        target = top[0]
        strategy_id = uuid.UUID(str(target["strategy_id"]))

        prompt = f"""
We want to optimize the parameters of this strategy:
Name: {target['name']}
Current Sharpe: {target['sharpe']}
Current composite score: {target['composite_score']}
Pair: {target['pair']}
Timeframe: {target['timeframe']}

Active insights from knowledge base:
{_format_insights(kb['active_insights'])}

Suggest a parameter optimization experiment. What parameters should we tune and in what direction?
Return JSON:
{{
  "rationale": "...",
  "prediction": "...",
  "parameter_hints": {{"param_name": "direction or range hint"}}
}}
"""
        result = await llm.complete_json(
            system="You are a quantitative trading strategy optimizer.",
            user=prompt,
        )

        # Headroom estimate: how much better could this strategy be?
        headroom = self._estimate_headroom(target)
        priority = headroom * 0.8 + 0.2  # min 0.2 priority

        return Hypothesis(
            id=uuid.uuid4(),
            type=HypothesisType.EXPLOIT,
            rationale=result.get("rationale", f"Optimizing parameters of {target['name']}"),
            prediction=result.get("prediction", ""),
            priority_score=priority,
            target_component_ids=[],
            target_strategy_id=strategy_id,
            parameter_changes=result.get("parameter_hints", {}),
            context={"pair": target["pair"], "timeframe": target["timeframe"]},
            status=HypothesisStatus.QUEUED,
        )

    async def _gen_ablate(self, kb: dict[str, Any]) -> Optional[Hypothesis]:
        """Schedule an ablation run to isolate a component's contribution."""
        components = await db.get_all_components()
        scores = await db.get_component_scores()

        # Find components with enough appearances but no ablation score yet
        needs_ablation = [
            s for s in scores
            if s["total_appearances"] >= 3 and s["total_ablations"] == 0
        ]

        if not needs_ablation:
            return None

        # Pick the component that appears most (highest signal)
        target = max(needs_ablation, key=lambda s: s["total_appearances"])
        target_component = next(
            (c for c in components if c["name"] == target["name"]), None
        )
        if not target_component:
            return None

        top = await db.get_top_strategies(limit=3)
        if not top:
            return None

        strategy_id = uuid.UUID(str(top[0]["strategy_id"]))
        priority = min(target["total_appearances"] * 0.1, 1.0)

        return Hypothesis(
            id=uuid.uuid4(),
            type=HypothesisType.ABLATE,
            rationale=(
                f"Component '{target['name']}' appeared in {target['total_appearances']} strategies "
                f"but has no ablation score. We need to isolate its marginal contribution."
            ),
            prediction="Removing this component will change Sharpe by a measurable amount",
            priority_score=priority,
            target_component_ids=[uuid.UUID(str(target_component["id"]))],
            target_strategy_id=strategy_id,
            parameter_changes={},
            context={"pair": top[0]["pair"], "timeframe": top[0]["timeframe"]},
            status=HypothesisStatus.QUEUED,
        )

    async def _gen_crossover(self, kb: dict[str, Any]) -> Optional[Hypothesis]:
        """Merge elements from two complementary strategies."""
        top = await db.get_top_strategies(limit=10)
        if len(top) < 2:
            return None

        prompt = f"""
We want to combine elements from two trading strategies.

Top strategies:
{_format_strategies(top)}

Active insights:
{_format_insights(kb['active_insights'])}

Select two strategies that are complementary (e.g., one has high win rate, other has low drawdown)
and explain what elements to merge.

Return JSON:
{{
  "strategy_a_name": "...",
  "strategy_b_name": "...",
  "rationale": "Why these two are complementary and what to merge",
  "prediction": "Expected improvement",
  "merge_approach": "Which entry/exit/filter from each parent to combine"
}}
"""
        result = await llm.complete_json(
            system="You are a quantitative trading strategy researcher.",
            user=prompt,
        )

        # Find IDs of selected strategies
        a = next((s for s in top if s["name"] == result.get("strategy_a_name")), top[0])
        b = next((s for s in top if s["name"] == result.get("strategy_b_name")), top[1])

        return Hypothesis(
            id=uuid.uuid4(),
            type=HypothesisType.CROSSOVER,
            rationale=result.get("rationale", "Crossover of two complementary strategies"),
            prediction=result.get("prediction", ""),
            priority_score=0.6,
            target_component_ids=[],
            target_strategy_id=uuid.UUID(str(a["strategy_id"])),
            parameter_changes={"secondary_strategy_id": str(b["strategy_id"]),
                               "merge_approach": result.get("merge_approach", "")},
            context={"pair": a["pair"], "timeframe": a["timeframe"]},
            status=HypothesisStatus.QUEUED,
        )

    async def _gen_stress_test(self, kb: dict[str, Any]) -> Optional[Hypothesis]:
        """Stress-test a high-scoring strategy across more contexts."""
        top = await db.get_top_strategies(limit=5)
        if not top:
            return None

        narrow = [s for s in top if kb["total_runs"] > 10]
        if not narrow:
            return None

        target = narrow[0]
        available_contexts = settings.available_contexts
        tested_contexts = {(c["pair"], c["timeframe"]) for c in kb["coverage"]}

        # Prefer genuinely untested contexts; fall back to all if none left
        untested = [
            ctx for ctx in available_contexts
            if (ctx["pair"], ctx["timeframe"]) not in tested_contexts
        ]
        contexts_to_test = untested[:3] if untested else available_contexts[:3]

        return Hypothesis(
            id=uuid.uuid4(),
            type=HypothesisType.STRESS_TEST,
            rationale=(
                f"Strategy '{target['name']}' scores {target['composite_score']:.3f} on "
                f"{target['pair']}/{target['timeframe']} but hasn't been tested broadly. "
                f"Testing across {len(contexts_to_test)} new contexts."
            ),
            prediction="Testing on more contexts will reveal if the strategy is robust or overfit",
            priority_score=0.65,
            target_component_ids=[],
            target_strategy_id=uuid.UUID(str(target["strategy_id"])),
            parameter_changes={"additional_contexts": contexts_to_test},
            context={"pair": target["pair"], "timeframe": target["timeframe"]},
            status=HypothesisStatus.QUEUED,
        )

    async def _gen_regime_specialize(self, kb: dict[str, Any]) -> Optional[Hypothesis]:
        """Add a regime filter to a high-variance strategy."""
        top = await db.get_top_strategies(limit=5)
        if not top:
            return None

        target = top[0]
        return Hypothesis(
            id=uuid.uuid4(),
            type=HypothesisType.REGIME_SPECIALIZE,
            rationale=(
                f"Adding a regime filter (e.g., ADX > 25 for trending markets) to "
                f"'{target['name']}' may improve performance in favorable regimes."
            ),
            prediction="Regime filter will reduce trades in unfavorable conditions and improve Sharpe",
            priority_score=0.55,
            target_component_ids=[],
            target_strategy_id=uuid.UUID(str(target["strategy_id"])),
            parameter_changes={"regime_type": "trend", "filter_threshold": 25},
            context={"pair": target["pair"], "timeframe": target["timeframe"]},
            status=HypothesisStatus.QUEUED,
        )

    def _bootstrap_explore(self, kb: dict[str, Any]) -> Hypothesis:
        """Very first hypothesis when the library is empty — use the first configured context."""
        first_context = settings.available_contexts[0]
        return Hypothesis(
            id=uuid.uuid4(),
            type=HypothesisType.EXPLORE,
            rationale="Bootstrapping: generating initial strategy from first principles.",
            prediction="Establish baseline performance for the knowledge base.",
            priority_score=1.0,
            target_component_ids=[],
            target_strategy_id=None,
            parameter_changes={},
            context=first_context,
            status=HypothesisStatus.QUEUED,
        )

    def _validate_context(
        self,
        chosen: dict,
        available: list[dict],
        coverage_map: dict[tuple, int],
    ) -> dict:
        """
        Validate the LLM-chosen context is in the configured pool.
        If not, fall back to the least-tested available context.
        """
        chosen_pair = chosen.get("pair", "")
        chosen_tf = chosen.get("timeframe", "")

        # Check if the chosen context is valid
        for ctx in available:
            if ctx["pair"] == chosen_pair and ctx["timeframe"] == chosen_tf:
                return ctx

        # Fall back: pick the least-tested context
        log.warning(
            "hypothesis.invalid_context",
            chosen=chosen,
            falling_back="least-tested available context",
        )
        return min(
            available,
            key=lambda ctx: coverage_map.get((ctx["pair"], ctx["timeframe"]), 0),
        )

    # --------------------------------------------------------
    # Priority helpers
    # --------------------------------------------------------

    def _priority_explore(
        self, kb: dict, score_map: dict, selected_names: list[str]
    ) -> float:
        if not selected_names:
            return 0.3
        scores = [score_map.get(n, 0) or 0 for n in selected_names]
        avg_score = sum(scores) / max(len(scores), 1)
        coverage_bonus = max(0, 1 - kb["total_runs"] / 100)  # explore more when data is sparse
        return min(avg_score * 0.5 + coverage_bonus * 0.5, 1.0)

    def _estimate_headroom(self, strategy: dict) -> float:
        """How much better could this strategy plausibly get?"""
        current = strategy.get("composite_score") or 0
        # Simple heuristic: headroom shrinks as score grows
        return max(0, 1.0 - current / 3.0)


# --------------------------------------------------------
# Formatting helpers for LLM prompts
# --------------------------------------------------------

def _format_components(components: list[dict], score_map: dict) -> str:
    lines = []
    for c in components[:30]:
        score = score_map.get(c["name"])
        score_str = f"{score:.3f}" if score is not None else "no score yet"
        lines.append(f"- [{c['category']}] {c['name']} (score: {score_str}): {c.get('description', '')[:80]}")
    return "\n".join(lines)


def _format_contexts(available: list[dict], coverage_map: dict[tuple, int]) -> str:
    lines = []
    for ctx in available:
        runs = coverage_map.get((ctx["pair"], ctx["timeframe"]), 0)
        lines.append(f"  - pair={ctx['pair']} timeframe={ctx['timeframe']} (runs so far: {runs})")
    return "\n".join(lines)


def _format_insights(insights: list[dict]) -> str:
    if not insights:
        return "  (none yet)"
    return "\n".join([
        f"  - [{i['insight_type']}] {i['statement']} (confidence: {i['confidence']:.2f})"
        for i in insights[:15]
    ])


def _format_strategies(strategies: list[dict]) -> str:
    lines = []
    for s in strategies:
        lines.append(
            f"- {s['name']} | pair={s['pair']} | tf={s['timeframe']} "
            f"| score={s['composite_score']:.3f} | sharpe={s['sharpe']:.2f} "
            f"| dd={s['max_drawdown']:.1%} | wr={s['win_rate']:.1%}"
        )
    return "\n".join(lines)


engine = HypothesisEngine()

"""
Insight Synthesizer — runs periodically to extract generalizable patterns
from accumulated backtest results and update the knowledge base.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any

import structlog

from src.db.database import db
from src.llm.client import llm
from src.models import Insight, InsightType

log = structlog.get_logger()


SYNTHESIS_SYSTEM = """
You are a quantitative research analyst reviewing algorithmic trading backtest results.
Your goal is to extract generalizable, falsifiable insights from experimental data.

Rules:
- Only state what the data actually supports — no speculation beyond the evidence
- Each insight must be falsifiable: "Component X improves metric Y when condition Z"
- Be specific about conditions (pair, timeframe, regime) when the data warrants it
- Distinguish correlation from causation: use "associated with" when causation is unclear
- Flag contradictions with existing insights
"""


class InsightSynthesizer:

    async def synthesize(self, recent_n: int = 20) -> list[Insight]:
        """
        Pull recent backtest runs and current insights, ask LLM to extract
        new patterns, save them to the DB, and update confidence of existing ones.
        """
        log.info("insight.synthesis.starting", recent_n=recent_n)

        # Gather evidence
        runs_summary = await self._fetch_recent_runs_summary(recent_n)
        existing_insights = await db.get_active_insights()
        component_scores = await db.get_component_scores()

        if not runs_summary:
            log.info("insight.synthesis.no_data")
            return []

        # Ask LLM to synthesize
        raw_insights = await self._call_llm(
            runs_summary, existing_insights, component_scores
        )

        # Persist new insights
        new_insights = []
        for raw in raw_insights.get("new_insights", []):
            insight = self._parse_insight(raw)
            if insight:
                await db.save_insight(insight)
                new_insights.append(insight)
                log.info(
                    "insight.saved",
                    type=insight.insight_type.value,
                    confidence=insight.confidence,
                    statement=insight.statement[:80],
                )

        # Update confidence of existing insights based on new evidence
        for update in raw_insights.get("confidence_updates", []):
            insight_id = self._find_insight_id(existing_insights, update.get("statement", ""))
            if insight_id:
                await db.update_insight_confidence(
                    insight_id=uuid.UUID(insight_id),
                    confidence=update["new_confidence"],
                    contradicted_by=[
                        uuid.UUID(r) for r in update.get("contradicted_by_run_ids", [])
                    ],
                )

        log.info(
            "insight.synthesis.complete",
            new_insights=len(new_insights),
            updates=len(raw_insights.get("confidence_updates", [])),
        )

        # Gap 5: produce actionable engine directives from this synthesis pass
        await self._produce_engine_directives(raw_insights, new_insights)

        return new_insights

    async def _produce_engine_directives(
        self,
        raw_insights: dict,
        new_insights: list,
    ) -> None:
        """
        After synthesis, ask the LLM to convert the emerging patterns into
        concrete sampling bias directives for the Hypothesis Engine.
        These replace the static weight table with data-driven priorities.
        """
        all_insights = await db.get_active_insights()
        failure_directives = await db.get_failure_directives(limit=15)

        if not all_insights:
            return

        directive_prompt = f"""
You are the strategic controller of an autonomous trading strategy research agent.
Based on the current state of the knowledge base, produce sampling directives that
tell the agent what experiment types to run more or less of.

=== CURRENT ACTIVE INSIGHTS ===
{self._format_existing_insights(all_insights[:20])}

=== ACTIVE FAILURE CONSTRAINTS ===
{chr(10).join(f"  [{d['root_cause']}] {d['constraint_text']}" for d in failure_directives[:10])}

=== SYNTHESIS SUMMARY FROM THIS PASS ===
{raw_insights.get('synthesis_summary', 'N/A')}

=== EXPERIMENT TYPES AVAILABLE ===
EXPLORE, EXPLOIT, ABLATE, CROSSOVER, STRESS_TEST, REGIME_SPECIALIZE

Produce 2-4 directives that will maximize learning efficiency given the current state.
Examples of good directives:
- "BOOST ABLATE if momentum components have high appearances but no ablation score"
- "SUPPRESS EXPLORE if failure_pattern insights are accumulating faster than positive ones"
- "BOOST EXPLOIT if top strategies haven't been parameter-tuned yet"
- "BOOST STRESS_TEST if best strategy has only been tested on 1 pair"
- "SUPPRESS CROSSOVER if we have fewer than 5 evaluated strategies"

Return JSON:
{{
  "directives": [
    {{
      "directive_type": "boost" | "suppress",
      "target_hypothesis_type": "EXPLORE|EXPLOIT|ABLATE|CROSSOVER|STRESS_TEST|REGIME_SPECIALIZE",
      "rationale": "Specific reason based on current KB state",
      "bias_weight": 0.1-0.5
    }}
  ],
  "strategic_summary": "One sentence: what phase is the agent in right now?"
}}
"""
        try:
            result = await llm.complete_json(
                system="You are the strategic controller of a trading research agent.",
                user=directive_prompt,
                temperature=0.1,
            )
            directives = result.get("directives", [])
            if directives:
                await db.save_engine_directives(directives)
                log.info(
                    "engine_directives.saved",
                    count=len(directives),
                    summary=result.get("strategic_summary", ""),
                )
        except Exception as e:
            log.error("engine_directives.error", error=str(e))

    async def _fetch_recent_runs_summary(self, n: int) -> list[dict[str, Any]]:
        """Build a structured summary of recent backtest results for the LLM."""
        async with db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    br.id, s.name as strategy_name, s.component_ids,
                    br.pair, br.timeframe, br.is_holdout,
                    br.regime_trend, br.regime_vol,
                    br.sharpe, br.sortino, br.max_drawdown, br.win_rate,
                    br.trade_count, br.composite_score, br.oos_sharpe_delta,
                    br.trade_count_stability, br.disqualified, br.disqualification_reason,
                    br.created_at
                FROM backtest_runs br
                JOIN strategies s ON s.id = br.strategy_id
                ORDER BY br.created_at DESC
                LIMIT $1
                """,
                n,
            )
            return [dict(r) for r in rows]

    async def _call_llm(
        self,
        runs: list[dict],
        existing_insights: list[dict],
        component_scores: list[dict],
    ) -> dict[str, Any]:
        runs_text = self._format_runs(runs)
        existing_text = self._format_existing_insights(existing_insights)
        components_text = self._format_component_scores(component_scores)

        prompt = f"""
Analyze these {len(runs)} recent freqtrade strategy backtest results and extract insights.

=== RECENT BACKTEST RESULTS ===
{runs_text}

=== COMPONENT PERFORMANCE SCORES ===
{components_text}

=== EXISTING INSIGHTS (for cross-reference) ===
{existing_text}

=== YOUR TASK ===
1. Identify patterns consistent across 3+ independent experiments
2. State each as a falsifiable insight with a confidence score
3. Flag if any new result contradicts an existing insight
4. Suggest confidence updates for existing insights based on new evidence

Return this exact JSON structure:
{{
  "new_insights": [
    {{
      "statement": "Specific falsifiable statement about what works or doesn't",
      "insight_type": "component_effect|regime_condition|parameter_sensitivity|interaction_effect|failure_pattern|robustness_signal",
      "confidence": 0.0-1.0,
      "evidence_run_ids": ["run_id_1", "run_id_2"],
      "reasoning": "Brief explanation of the evidence"
    }}
  ],
  "confidence_updates": [
    {{
      "statement": "Statement of an existing insight to update",
      "new_confidence": 0.0-1.0,
      "contradicted_by_run_ids": [],
      "reasoning": "Why confidence changed"
    }}
  ],
  "synthesis_summary": "2-3 sentence summary of the most important patterns emerging"
}}
"""
        return await llm.complete_json(
            system=SYNTHESIS_SYSTEM,
            user=prompt,
            max_tokens=4096,
            temperature=0.1,
        )

    def _parse_insight(self, raw: dict) -> Insight | None:
        try:
            insight_type_map = {t.value: t for t in InsightType}
            insight_type = insight_type_map.get(
                raw.get("insight_type", ""), InsightType.COMPONENT_EFFECT
            )
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
            evidence_ids = []
            for rid in raw.get("evidence_run_ids", []):
                try:
                    evidence_ids.append(uuid.UUID(str(rid)))
                except Exception:
                    pass

            return Insight(
                id=uuid.uuid4(),
                statement=raw["statement"],
                insight_type=insight_type,
                confidence=confidence,
                evidence_run_ids=evidence_ids,
                contradicted_by=[],
                active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )
        except Exception as e:
            log.warning("insight.parse_error", error=str(e), raw=raw)
            return None

    def _find_insight_id(self, insights: list[dict], statement: str) -> str | None:
        """Find an existing insight by approximate statement match."""
        statement_lower = statement.lower()[:50]
        for i in insights:
            if statement_lower in i["statement"].lower():
                return str(i["id"])
        return None

    # --------------------------------------------------------
    # Formatting helpers
    # --------------------------------------------------------

    def _format_runs(self, runs: list[dict]) -> str:
        lines = []
        for r in runs:
            status = "DISQUALIFIED" if r["disqualified"] else "OK"
            lines.append(
                f"[{r['id']}] {r['strategy_name']} | {r['pair']} {r['timeframe']} | "
                f"regime={r['regime_trend']}/{r['regime_vol']} | "
                f"sharpe={r['sharpe']} | dd={r['max_drawdown']} | "
                f"trades={r['trade_count']} | score={r['composite_score']} | "
                f"oos_delta={r['oos_sharpe_delta']} | {status}"
                + (f" ({r['disqualification_reason']})" if r.get("disqualification_reason") else "")
            )
        return "\n".join(lines)

    def _format_existing_insights(self, insights: list[dict]) -> str:
        if not insights:
            return "  (none yet)"
        return "\n".join([
            f"  [{i['insight_type']}] (conf={i['confidence']:.2f}) {i['statement']}"
            for i in insights[:20]
        ])

    def _format_component_scores(self, scores: list[dict]) -> str:
        if not scores:
            return "  (no scores yet)"
        return "\n".join([
            f"  {s['name']} | category={s.get('category','')} | "
            f"avg_sharpe_contrib={s['avg_sharpe_contrib']} | appearances={s['total_appearances']}"
            for s in scores[:20]
        ])


synthesizer = InsightSynthesizer()

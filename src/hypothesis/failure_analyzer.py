"""
Failure Analyzer — extracts structured failure patterns from REFUTED experiments.

This is the key self-learning mechanism: instead of simply discarding failed
strategies, the agent dissects WHY they failed and stores structured negative
insights that directly constrain future generation.

Failure patterns are stored as insights with type='failure_pattern' and
negative confidence signals, plus a structured `failure_directives` table
that the Hypothesis Engine reads to bias its sampling.
"""
from __future__ import annotations
import ast
import uuid
from datetime import datetime
from typing import Any

import structlog

from src.db.database import db
from src.llm.client import llm
from src.models import Insight, InsightType

log = structlog.get_logger()


FAILURE_ANALYSIS_SYSTEM = """
You are a quantitative trading strategy post-mortem analyst.
Your job is to extract precise, actionable failure patterns from failed backtests.

Rules:
- Be specific: name the exact indicators, regimes, and conditions that caused failure
- Distinguish root causes: too few trades vs high drawdown vs OOS degradation vs walk-forward fragility
- Phrase each pattern as a constraint for future generation: "Avoid X when Y"
- Separate structural failures (code/logic) from market failures (signal doesn't work in this regime)
- Your output is consumed by an LLM code generator — be concise and machine-readable
"""


class FailureAnalyzer:

    async def analyze(
        self,
        strategy_code: str,
        metrics: dict[str, Any],
        disqualification_reason: str | None,
        outcome_summary: str,
        context: dict[str, Any],
        hypothesis_type: str,
        run_ids: list[uuid.UUID],
    ) -> list[Insight]:
        """
        Analyze a failed experiment and extract structured failure patterns.
        Returns a list of negative Insight objects saved to the KB.
        """
        pair = context.get("pair", "unknown")
        timeframe = context.get("timeframe", "unknown")

        log.info(
            "failure_analyzer.starting",
            pair=pair,
            timeframe=timeframe,
            reason=disqualification_reason,
        )

        syntax_issue_note = self._detect_syntax_issue(strategy_code)
        strategy_excerpt = self._build_strategy_excerpt(strategy_code)

        prompt = f"""
Analyze this failed Freqtrade strategy experiment and extract failure patterns.

=== CONTEXT ===
Pair: {pair}
Timeframe: {timeframe}
Hypothesis type: {hypothesis_type}
Disqualification reason: {disqualification_reason or "none — strategy ran but scored poorly"}
Outcome summary: {outcome_summary}

=== METRICS ===
Sharpe: {metrics.get('sharpe')}
Sortino: {metrics.get('sortino')}
Max drawdown: {metrics.get('max_drawdown')}
Win rate: {metrics.get('win_rate')}
Trade count: {metrics.get('trade_count')}
Avg trade duration (h): {metrics.get('avg_trade_dur_h')}
Profit factor: {metrics.get('profit_factor')}
OOS Sharpe delta: {metrics.get('oos_sharpe_delta')}
Trade count stability (CV): {metrics.get('trade_count_stability')}

=== STRATEGY CODE ===
{strategy_excerpt}

Note: The strategy code excerpt may be abbreviated for token efficiency.
Do not claim the file is truncated/incomplete solely because the excerpt ends.

=== YOUR TASK ===
Identify 1-3 specific failure patterns. For each, determine:

1. The ROOT CAUSE category:
   - "insufficient_trades": signal fires too rarely
   - "high_drawdown": entries at wrong time, poor risk management
   - "oos_degradation": in-sample overfitting, signal doesn't generalize
   - "walk_forward_fragility": performance collapsed across time periods
   - "signal_noise": entry signal fires on noise, low precision
   - "exit_timing": poor exit logic causing giving back gains
   - "regime_mismatch": strategy suited for a different market regime

2. A specific, falsifiable FAILURE STATEMENT:
   e.g. "RSI(14) < 30 entry on BTC/4h generates < 20 trades/month in low-volatility regimes"
   e.g. "MACD crossover entry with BB filter causes > 25% drawdown in trending BTC markets"

3. A CONSTRAINT for future code generation:
   e.g. "Do not use RSI oversold as primary entry on BTC/4h without a volume confirmation filter"
   e.g. "Avoid MACD as entry signal on trending pairs without a trend strength gate (ADX > 25)"

4. Whether this is CONTEXT-SPECIFIC (only this pair/timeframe) or GENERAL (any context)

Return JSON:
{{
  "failure_patterns": [
    {{
      "root_cause": "...",
      "statement": "Specific falsifiable failure statement",
      "constraint": "Avoid/Do not X when Y — directive for code generator",
      "scope": "context_specific" | "general",
      "confidence": 0.5-0.9,
      "indicators_involved": ["rsi", "macd", ...]
    }}
  ],
  "structural_issue": true | false,
  "structural_note": "If true: describe the code/logic flaw found"
}}
"""
        try:
            result = await llm.complete_json(
                system=FAILURE_ANALYSIS_SYSTEM,
                user=prompt,
                temperature=0.1,
            )
        except Exception as e:
            log.error("failure_analyzer.llm_error", error=str(e))
            return []

        insights = []
        for fp in result.get("failure_patterns", []):
            statement = fp.get("statement", "")
            constraint = fp.get("constraint", "")
            if not statement:
                continue

            # Combine statement + constraint into a single actionable insight
            full_statement = f"[FAILURE] {statement} | CONSTRAINT: {constraint}"
            if fp.get("scope") == "context_specific":
                full_statement = f"[{pair}/{timeframe}] {full_statement}"

            insight = Insight(
                id=uuid.uuid4(),
                statement=full_statement,
                insight_type=InsightType.FAILURE_PATTERN,
                confidence=float(fp.get("confidence", 0.6)),
                evidence_run_ids=run_ids,
                contradicted_by=[],
                active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )
            await db.save_insight(insight)
            insights.append(insight)

            # Also save a structured directive for the Hypothesis Engine
            await db.save_failure_directive({
                "constraint": constraint,
                "root_cause": fp.get("root_cause", "unknown"),
                "scope": fp.get("scope", "general"),
                "pair": pair if fp.get("scope") == "context_specific" else None,
                "timeframe": timeframe if fp.get("scope") == "context_specific" else None,
                "indicators_involved": fp.get("indicators_involved", []),
                "confidence": fp.get("confidence", 0.6),
                "evidence_insight_id": str(insight.id),
            })

            log.info(
                "failure_analyzer.pattern_saved",
                root_cause=fp.get("root_cause"),
                statement=statement[:80],
            )

        # Handle structural code issues
        structural_note = None
        if syntax_issue_note:
            structural_note = syntax_issue_note
        elif result.get("structural_issue") and result.get("structural_note"):
            candidate = str(result.get("structural_note", "")).strip()
            if candidate and not self._looks_like_truncation_false_positive(candidate):
                structural_note = candidate

        if structural_note:
            structural_insight = Insight(
                id=uuid.uuid4(),
                statement=f"[STRUCTURAL] {structural_note}",
                insight_type=InsightType.FAILURE_PATTERN,
                confidence=0.8,  # structural issues are high confidence
                evidence_run_ids=run_ids,
                contradicted_by=[],
                active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )
            await db.save_insight(structural_insight)
            insights.append(structural_insight)

        log.info("failure_analyzer.complete", patterns_found=len(insights))
        return insights

    def _build_strategy_excerpt(self, strategy_code: str) -> str:
        max_chars = 7000
        if len(strategy_code) <= max_chars:
            return strategy_code

        head = strategy_code[:4200]
        tail = strategy_code[-2400:]
        marker = "\n\n# --- EXCERPT TRUNCATED FOR ANALYSIS ---\n\n"
        return head + marker + tail

    def _detect_syntax_issue(self, strategy_code: str) -> str | None:
        try:
            ast.parse(strategy_code)
            return None
        except SyntaxError as e:
            return f"Python syntax error at line {e.lineno}: {e.msg}"

    def _looks_like_truncation_false_positive(self, note: str) -> bool:
        lowered = note.lower()
        indicators = [
            "truncat",
            "incomplete",
            "cut off",
            "ends abruptly",
            "missing end",
        ]
        return any(tok in lowered for tok in indicators)


analyzer = FailureAnalyzer()

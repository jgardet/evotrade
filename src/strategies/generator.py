"""
Strategy Generator — uses the LLM to write valid Freqtrade strategy Python files.
Takes a hypothesis + selected components and produces runnable code.
"""
from __future__ import annotations
import uuid
import textwrap
from datetime import datetime
from typing import Any

import structlog

from src.config import settings
from src.llm.client import llm
from src.models import (
    Component, ComponentCategory, ComponentOrigin,
    Hypothesis, HypothesisType, Strategy, StrategyStatus,
)

log = structlog.get_logger()


STRATEGY_SYSTEM_PROMPT = """
You are an expert algorithmic trading developer specializing in Freqtrade strategies.
You write clean, correct, production-ready Python code that is compatible with Freqtrade's
IStrategy interface. Your strategies are testable, well-commented, and avoid common pitfalls
like lookahead bias.

Rules:
- Always inherit from IStrategy
- Always define: minimal_roi, stoploss, timeframe, populate_indicators, populate_entry_trend, populate_exit_trend
- Never use future data (no shift(-1) in signals)
- Use pandas_ta or talib for indicators
- Parameters defined as class-level IntParameter / DecimalParameter for optimization
- Return the dataframe with a 'enter_long' or 'exit_long' column (boolean)
- Keep code clean and commented
"""


COMPONENT_SYSTEM_PROMPT = """
You are an expert algorithmic trading developer. You write self-contained Python code snippets
for Freqtrade strategy components. Each snippet is a method body or utility block that can be
composed into a full IStrategy class. Snippets must:
- Use only pandas, numpy, pandas_ta, or talib
- Accept a `dataframe` argument and return it modified
- Be clearly commented
- Have no side effects
"""


class StrategyGenerator:

    async def generate_from_hypothesis(
        self,
        hypothesis: Hypothesis,
        components: list[Component],
        existing_insights: list[dict],
    ) -> Strategy:
        """
        Main entry point. Generates a Strategy object from a hypothesis and
        selected components. The LLM writes the actual Python code.
        """
        log.info(
            "strategy.generating",
            hypothesis_type=hypothesis.type.value,
            n_components=len(components),
        )

        strategy_name = self._make_name(hypothesis)
        code = await self._generate_code(
            hypothesis, components, existing_insights, strategy_name
        )

        strategy = Strategy(
            id=uuid.uuid4(),
            name=strategy_name,
            code=code,
            component_ids=[c.id for c in components],
            parameters=self._extract_parameters(components),
            hypothesis_id=hypothesis.id,
            parent_ids=([hypothesis.target_strategy_id]
                        if hypothesis.target_strategy_id else []),
            generation=0,  # caller sets this from parent
            status=StrategyStatus.CREATED,
        )

        log.info("strategy.generated", strategy_id=str(strategy.id), name=strategy_name)
        return strategy

    async def generate_component(
        self,
        category: ComponentCategory,
        description: str,
        context: dict[str, Any],
    ) -> Component:
        """Ask the LLM to generate a new atomic component."""
        prompt = f"""
Generate a Freqtrade strategy component with the following specification:

Category: {category.value}
Description: {description}
Context: {context}

Return a JSON object with these fields:
- name: snake_case identifier (unique, descriptive)
- code_snippet: the Python code block (method body, starts with `def ...`)
- parameters: dict of default parameter values
- parameter_space: dict mapping param names to [min, max] or [list of choices]
- dependencies: list of required libraries (e.g. ["pandas_ta", "talib"])
- description: one-sentence explanation
"""
        result = await llm.complete_json(
            system=COMPONENT_SYSTEM_PROMPT,
            user=prompt,
        )

        return Component(
            id=uuid.uuid4(),
            category=category,
            name=result["name"],
            code_snippet=result["code_snippet"],
            parameters=result.get("parameters", {}),
            parameter_space=result.get("parameter_space", {}),
            dependencies=result.get("dependencies", []),
            description=result.get("description", description),
            origin=ComponentOrigin.LLM_GENERATED,
        )

    async def mutate_component(
        self,
        component: Component,
        mutation_instruction: str,
    ) -> Component:
        """Create a mutated variant of an existing component."""
        prompt = f"""
Mutate this existing Freqtrade strategy component.

Original component:
Name: {component.name}
Category: {component.category.value}
Code:
{component.code_snippet}
Parameters: {component.parameters}

Mutation instruction: {mutation_instruction}

Return a JSON object with the same fields as before, but with your changes applied.
Give it a new descriptive name that reflects the mutation.
"""
        result = await llm.complete_json(
            system=COMPONENT_SYSTEM_PROMPT,
            user=prompt,
        )

        return Component(
            id=uuid.uuid4(),
            category=component.category,
            name=result["name"],
            code_snippet=result["code_snippet"],
            parameters=result.get("parameters", component.parameters),
            parameter_space=result.get("parameter_space", component.parameter_space),
            dependencies=result.get("dependencies", component.dependencies),
            description=result.get("description", ""),
            origin=ComponentOrigin.MUTATION,
            parent_id=component.id,
        )

    async def _generate_code(
        self,
        hypothesis: Hypothesis,
        components: list[Component],
        insights: list[dict],
        strategy_name: str,
    ) -> str:
        components_text = "\n\n".join([
            f"--- {c.category.value}: {c.name} ---\n{c.code_snippet}"
            for c in components
        ])

        insights_text = "\n".join([
            f"- [{i['insight_type']}] (confidence {i['confidence']:.2f}): {i['statement']}"
            for i in insights[:15]  # top 15 insights
        ])

        prompt = f"""
Generate a complete Freqtrade strategy class named `{strategy_name}`.

Hypothesis being tested:
Type: {hypothesis.type.value}
Rationale: {hypothesis.rationale}
Prediction: {hypothesis.prediction}

Components to compose into this strategy:
{components_text}

Relevant insights from the knowledge base (follow these!):
{insights_text if insights_text else "No insights yet — this is an early exploration."}

Context:
{hypothesis.context}

Requirements:
1. Compose all provided components coherently — they should work together logically
2. Follow the insights from the knowledge base (avoid known failure patterns)
3. Use IntParameter / DecimalParameter for all numeric thresholds to enable optimization
4. Add a docstring explaining the strategy logic
5. Set reasonable default minimal_roi, stoploss, and timeframe values based on the context
6. Return complete, runnable Python code only — no markdown, no explanation outside comments
"""

        code = await llm.complete(
            system=STRATEGY_SYSTEM_PROMPT,
            user=prompt,
            max_tokens=6000,
            temperature=0.15,
        )

        # Strip markdown code fences if present
        code = code.strip()
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:])
        if code.endswith("```"):
            code = "\n".join(code.split("\n")[:-1])

        return code.strip()

    def _make_name(self, hypothesis: Hypothesis) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"Agent_{hypothesis.type.value}_{ts}"

    def _extract_parameters(self, components: list[Component]) -> dict[str, Any]:
        merged = {}
        for c in components:
            merged.update(c.parameters)
        return merged


generator = StrategyGenerator()

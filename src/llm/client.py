"""
LLM client abstraction.
Supports Anthropic Claude and OpenAI (Codex/GPT-4o).
All calls are async and return structured text.
"""
from __future__ import annotations
import json
import re
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

log = structlog.get_logger()


class LLMClient:
    """
    Unified async LLM client.
    Backend is selected at startup from settings.llm_backend.
    """

    def __init__(self):
        self.backend = settings.llm_backend
        self._anthropic = None
        self._openai = None

    def _get_anthropic(self):
        if self._anthropic is None:
            import anthropic
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key
            )
        return self._anthropic

    def _get_openai(self):
        if self._openai is None:
            from openai import AsyncOpenAI
            self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._openai

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str:
        """Call the configured LLM and return the text response."""
        log.debug("llm.request", backend=self.backend, user_preview=user[:80])

        if self.backend == "claude":
            return await self._complete_claude(system, user, max_tokens, temperature)
        else:
            return await self._complete_openai(system, user, max_tokens, temperature)

    async def _complete_claude(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        client = self._get_anthropic()
        response = await client.messages.create(
            model=settings.claude_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    async def _complete_openai(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        client = self._get_openai()
        payload = {
            "model": settings.openai_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        try:
            response = await client.chat.completions.create(
                max_completion_tokens=max_tokens,
                **payload,
            )
        except Exception as e:
            msg = str(e)
            if "max_completion_tokens" not in msg and "unsupported_parameter" not in msg:
                raise
            response = await client.chat.completions.create(
                max_tokens=max_tokens,
                **payload,
            )
        return response.choices[0].message.content

    async def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> Any:
        """Call LLM expecting a JSON response. Strips markdown fences and parses."""
        raw = await self.complete(
            system=system + "\n\nRespond ONLY with valid JSON. No markdown, no explanation.",
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._parse_json(raw)

    def _parse_json(self, raw: str) -> Any:
        # Strip ```json ... ``` fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            log.error("llm.json_parse_error", raw=raw[:200], error=str(e))
            raise


# Singleton
llm = LLMClient()

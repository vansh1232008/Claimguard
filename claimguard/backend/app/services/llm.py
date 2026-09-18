"""LLM access layer.

Two backends behind one interface:

* ``GeminiBackend``  - real Gemini calls over the REST API (needs GEMINI_API_KEY).
* ``OfflineBackend`` - a deterministic rule-based reasoner that returns the same
  JSON shapes. It keeps the whole pipeline runnable (and testable, and
  reproducible) with no network and no keys.

Every agent asks for JSON with an explicit schema, so a swap between the two is
invisible to the agents.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.core.logging import get_logger
from app.services.stub_reasoner import STUB_HANDLERS

logger = get_logger(__name__)

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass
class LLMResponse:
    text: str
    data: dict[str, Any] = field(default_factory=dict)
    model: str = "offline-stub"
    latency_ms: int = 0
    stub: bool = True


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


class OfflineBackend:
    """Deterministic stand-in for the LLM.

    Not a mock that returns fixed strings: each task has a small rule-based
    reasoner that actually looks at the claim context, so the pipeline produces
    meaningful (and stable) output without any API key.
    """

    name = "offline-stub"

    def complete_json(self, task: str, context: dict[str, Any], **_: Any) -> LLMResponse:
        started = time.perf_counter()
        handler = STUB_HANDLERS.get(task)
        if handler is None:
            logger.warning("No offline handler for task %r - returning empty object", task)
            data: dict[str, Any] = {}
        else:
            data = handler(context)
        elapsed = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            text=json.dumps(data), data=data, model=self.name, latency_ms=elapsed, stub=True
        )


class GeminiBackend:
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete_json(
        self,
        task: str,
        context: dict[str, Any],
        *,
        instruction: str = "",
        schema: dict[str, Any] | None = None,
        **_: Any,
    ) -> LLMResponse:
        import httpx  # local import: only needed on the online path

        prompt = build_prompt(task, context, instruction, schema)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": settings.llm_temperature,
                "maxOutputTokens": settings.llm_max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        url = GEMINI_ENDPOINT.format(model=self.model)
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(url, params={"key": self.api_key}, json=payload)
                resp.raise_for_status()
                body = resp.json()
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:  # network down, quota, malformed - degrade, never crash
            logger.warning("Gemini call failed for task %r (%s); falling back offline", task, exc)
            return OfflineBackend().complete_json(task, context)
        elapsed = int((time.perf_counter() - started) * 1000)
        data = _extract_json(text)
        if not data:
            logger.warning("Gemini returned unparseable JSON for %r; falling back offline", task)
            return OfflineBackend().complete_json(task, context)
        return LLMResponse(text=text, data=data, model=self.model, latency_ms=elapsed, stub=False)


def build_prompt(
    task: str, context: dict[str, Any], instruction: str, schema: dict[str, Any] | None
) -> str:
    parts = [
        "You are an expert insurance fraud investigator working inside an automated "
        "claims adjudication pipeline. You are precise, evidence-driven and never "
        "invent facts that are not in the supplied context.",
        f"\n## Task\n{task}",
    ]
    if instruction:
        parts.append(f"\n## Instructions\n{instruction}")
    parts.append("\n## Context\n" + json.dumps(context, indent=2, default=str))
    if schema:
        parts.append(
            "\n## Output\nReturn ONLY a JSON object matching this schema:\n"
            + json.dumps(schema, indent=2)
        )
    parts.append(
        "\nGround every claim you make in the context. If evidence is missing, say so "
        "explicitly rather than assuming."
    )
    return "\n".join(parts)


class LLMService:
    """Thin façade the agents use. Counts calls so we can report cost per claim."""

    def __init__(self) -> None:
        if settings.gemini_api_key:
            self.backend: Any = GeminiBackend(settings.gemini_api_key, settings.gemini_model)
        else:
            self.backend = OfflineBackend()
        self.call_count = 0
        logger.info("LLM backend: %s", self.backend.name)

    @property
    def mode(self) -> str:
        return self.backend.name

    def complete_json(
        self,
        task: str,
        context: dict[str, Any],
        *,
        instruction: str = "",
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        return self.backend.complete_json(
            task, context, instruction=instruction, schema=schema
        )


_llm_singleton: LLMService | None = None


def get_llm() -> LLMService:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMService()
    return _llm_singleton

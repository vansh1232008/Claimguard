"""Base class for all ClaimGuard agents.

Gives every agent the same three guarantees:
  * it is timed and its LLM usage is counted,
  * a failure inside it degrades the investigation instead of killing it,
  * its output lands in the state in one consistent shape (the audit trail).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from app.agents.state import AgentResult, InvestigationState
from app.core.logging import get_logger
from app.services.llm import get_llm


class BaseAgent(ABC):
    name: str = "agent"
    description: str = ""
    # When True the orchestrator aborts the run if this agent fails.
    critical: bool = False

    def __init__(self) -> None:
        self.logger = get_logger(f"agent.{self.name}")
        self.llm = get_llm()

    # -- to implement -----------------------------------------------------
    @abstractmethod
    def run(self, state: InvestigationState) -> AgentResult:
        """Do the work. Raise freely - __call__ catches and records."""

    # -- orchestration glue ----------------------------------------------
    def __call__(self, state: InvestigationState) -> InvestigationState:
        started = time.perf_counter()
        llm_before = self.llm.call_count
        try:
            result = self.run(state)
            result.status = result.status or "ok"
        except Exception as exc:  # noqa: BLE001 - deliberate: agents degrade
            self.logger.exception("Agent %s failed", self.name)
            result = AgentResult(
                name=self.name,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                summary=f"{self.name} could not complete; downstream agents continued without it.",
            )
            state.setdefault("errors", []).append(f"{self.name}: {exc}")

        result.duration_ms = int((time.perf_counter() - started) * 1000)
        result.llm_calls = self.llm.call_count - llm_before
        return self.merge(state, result)

    def merge(self, state: InvestigationState, result: AgentResult) -> InvestigationState:
        state.setdefault("agent_runs", []).append(
            {
                "agent_name": result.name,
                "status": result.status,
                "summary": result.summary,
                "output": result.output,
                "risk_contribution": result.risk_contribution,
                "llm_calls": result.llm_calls,
                "duration_ms": result.duration_ms,
                "error": result.error,
            }
        )
        if result.signals:
            state.setdefault("risk_signals", []).extend(result.signals)
        if result.evidence:
            state.setdefault("evidence", []).extend(result.evidence)
        state["llm_calls"] = state.get("llm_calls", 0) + result.llm_calls
        if result.output:
            state[self.state_key] = result.output  # type: ignore[literal-required]
        return state

    @property
    def state_key(self) -> str:
        return self.name

    # -- helpers ----------------------------------------------------------
    def ask(
        self,
        task: str,
        context: dict[str, Any],
        *,
        instruction: str = "",
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.llm.complete_json(
            task, context, instruction=instruction, schema=schema
        ).data

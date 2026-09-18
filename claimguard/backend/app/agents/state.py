"""Shared state passed between agents.

LangGraph passes a plain dict between nodes; this module defines the contract
for that dict so every agent knows exactly what it can read and what it is
expected to write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class RiskSignal(TypedDict, total=False):
    source: str          # which agent raised it
    code: str            # machine-readable signal code
    detail: str          # one sentence an investigator can read
    severity: str        # low | medium | high
    weight: float        # 0..1 contribution to the blended fraud score


class InvestigationState(TypedDict, total=False):
    # inputs
    claim_id: str
    claim: dict[str, Any]
    policy: dict[str, Any]
    customer: dict[str, Any]
    vehicle: dict[str, Any]
    vendor: dict[str, Any]
    documents: list[dict[str, Any]]
    image_paths: list[str]
    options: dict[str, Any]

    # agent outputs
    intake: dict[str, Any]
    document_analysis: dict[str, Any]
    damage_analysis: dict[str, Any]
    coverage: dict[str, Any]
    anomaly: dict[str, Any]
    network: dict[str, Any]
    adjudication: dict[str, Any]

    # accumulators
    risk_signals: list[RiskSignal]
    evidence: list[dict[str, Any]]
    agent_runs: list[dict[str, Any]]
    fraud_score: float
    errors: list[str]
    llm_calls: int


@dataclass
class AgentResult:
    """What every agent hands back to the orchestrator."""

    name: str
    output: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    signals: list[RiskSignal] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    risk_contribution: float = 0.0
    llm_calls: int = 0
    duration_ms: int = 0
    status: str = "ok"
    error: str | None = None


def new_state(claim_id: str, **kwargs: Any) -> InvestigationState:
    state: InvestigationState = {
        "claim_id": claim_id,
        "risk_signals": [],
        "evidence": [],
        "agent_runs": [],
        "errors": [],
        "fraud_score": 0.0,
        "llm_calls": 0,
        "documents": [],
        "image_paths": [],
        "options": {},
    }
    state.update(kwargs)  # type: ignore[typeddict-item]
    return state


def signal(
    source: str, code: str, detail: str, severity: str = "medium", weight: float = 0.1
) -> RiskSignal:
    return {
        "source": source,
        "code": code,
        "detail": detail,
        "severity": severity,
        "weight": round(float(weight), 4),
    }

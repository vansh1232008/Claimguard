"""Agent 7 — Adjudication.

Blends the statistical score with the rule-derived signals, then produces the
approve / review / reject recommendation with a written rationale and the
evidence it rests on. This is the only agent allowed to output a decision.

Blending rule (deliberately simple and auditable):

    fraud_score = 0.65 * model_probability + 0.35 * signal_pressure

``signal_pressure`` is the summed weight of every signal raised upstream,
squashed into 0..1. Keeping it linear means an investigator can reconstruct the
number by hand from the signal list, which matters when a decline is disputed.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.agents.state import AgentResult, InvestigationState
from app.config import settings

SCHEMA: dict[str, Any] = {
    "decision": "approve|review|reject",
    "confidence": "number between 0 and 1",
    "rationale": "string",
    "recommended_payout": "number",
    "key_evidence": ["string"],
    "next_actions": ["string"],
}

INSTRUCTION = (
    "Make the final recommendation on this claim. Weigh coverage first (an uncovered "
    "loss cannot be paid regardless of fraud score), then the fraud score, then the "
    "corroborating signals. Write a rationale an investigator could send to the "
    "policyholder: specific, evidence-backed, no hedging."
)

MODEL_WEIGHT = 0.65
SIGNAL_WEIGHT = 0.35


def blend_score(model_probability: float, signals: list[dict[str, Any]]) -> tuple[float, float]:
    pressure = sum(float(s.get("weight", 0.0)) for s in signals)
    # Squash: 0 -> 0, 0.5 -> ~0.5, 1.5+ -> ~0.9
    signal_pressure = pressure / (pressure + 0.75) if pressure > 0 else 0.0
    score = MODEL_WEIGHT * float(model_probability) + SIGNAL_WEIGHT * signal_pressure
    return round(min(1.0, max(0.0, score)), 4), round(signal_pressure, 4)


class DecisionAgent(BaseAgent):
    name = "adjudication"
    state_key = "adjudication"
    description = "Blends every signal into an explainable recommendation"
    critical = True

    def run(self, state: InvestigationState) -> AgentResult:
        anomaly = state.get("anomaly", {}) or {}
        coverage = state.get("coverage", {}) or {}
        signals = state.get("risk_signals", []) or []

        model_prob = float(anomaly.get("fraud_probability", 0.0))
        fraud_score, signal_pressure = blend_score(model_prob, signals)
        state["fraud_score"] = fraud_score

        data = self.ask(
            "final_adjudication",
            {
                "claim": state.get("claim", {}),
                "fraud_score": fraud_score,
                "model_probability": model_prob,
                "signal_pressure": signal_pressure,
                "coverage": coverage,
                "risk_signals": signals,
                "document_analysis": state.get("document_analysis", {}),
                "damage_analysis": state.get("damage_analysis", {}),
                "network": state.get("network", {}),
                "auto_approve_below": settings.auto_approve_below,
                "auto_reject_above": settings.auto_reject_above,
            },
            instruction=INSTRUCTION,
            schema=SCHEMA,
        )

        decision = str(data.get("decision", "review")).lower()
        if decision not in {"approve", "review", "reject"}:
            decision = "review"

        # Guardrails the LLM cannot override.
        if not coverage.get("covered", True):
            decision = "reject"
            data["recommended_payout"] = 0.0
        elif (
            decision == "approve"
            and float(state.get("claim", {}).get("claimed_amount") or 0)
            > settings.high_value_claim_threshold
        ):
            # High-value claims never auto-approve, whatever the score says.
            decision = "review"
            data["rationale"] = (
                str(data.get("rationale", ""))
                + " Routed to manual review because the claim exceeds the "
                f"{settings.high_value_claim_threshold:,.0f} auto-approval ceiling."
            )

        payout = float(data.get("recommended_payout") or 0.0)
        if decision == "reject":
            payout = 0.0

        return AgentResult(
            name=self.name,
            output={
                **data,
                "decision": decision,
                "recommended_payout": round(payout, 2),
                "fraud_score": fraud_score,
                "model_probability": model_prob,
                "signal_pressure": signal_pressure,
                "thresholds": {
                    "auto_approve_below": settings.auto_approve_below,
                    "auto_reject_above": settings.auto_reject_above,
                },
            },
            summary=data.get("rationale", "Decision recorded."),
            risk_contribution=0.0,
        )

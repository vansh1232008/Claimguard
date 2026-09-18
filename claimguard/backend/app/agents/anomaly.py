"""Agent 5 — Statistical anomaly detection.

Runs the supervised fraud model (XGBoost, with GraphSAGE graph features when
they are available) and then asks the LLM to turn the feature contributions
into an explanation an investigator can act on. The score is the model's; the
words are the model's reasons, not a guess.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.agents.state import AgentResult, InvestigationState, signal
from app.services.graph_sync import graph_features_for_claim
from app.services.ml import get_scorer

SCHEMA: dict[str, Any] = {
    "severity": "low|moderate|high|critical",
    "drivers": [
        {"feature": "string", "value": "number", "effect": "raises|lowers", "weight": "number"}
    ],
    "explanation": "string",
}

INSTRUCTION = (
    "Explain, in two or three plain sentences an investigator can put in a file note, "
    "why the model scored this claim the way it did. Name the features that drove the "
    "score and say which way each pushed it. Do not invent features."
)


class AnomalyAgent(BaseAgent):
    name = "anomaly"
    state_key = "anomaly"
    description = "Gradient-boosted fraud scoring with feature attribution"

    def run(self, state: InvestigationState) -> AgentResult:
        claim = dict(state.get("claim", {}))
        policy = state.get("policy", {})
        customer = state.get("customer", {})
        vehicle = state.get("vehicle", {})
        vendor = state.get("vendor", {}) or {}
        docs = state.get("documents", []) or []
        doc_analysis = state.get("document_analysis", {}) or {}

        graph_feats = graph_features_for_claim(state["claim_id"])
        state["_graph_context"] = {  # type: ignore[typeddict-unknown-key]
            "ring": graph_feats.pop("_ring", {}),
            "neighbours": graph_feats.pop("_neighbours", []),
        }

        ctx: dict[str, Any] = {
            **claim,
            "coverage_limit": policy.get("coverage_limit"),
            "annual_premium": policy.get("annual_premium"),
            "deductible": policy.get("deductible"),
            "policy_start_date": policy.get("start_date"),
            "bonus_malus": policy.get("bonus_malus"),
            "customer_tenure_months": customer.get("tenure_months"),
            "vehicle_age": vehicle.get("age"),
            "vehicle_power": vehicle.get("vehicle_power"),
            "document_count": len(docs),
            "has_invoice": any(d.get("doc_type") == "invoice" for d in docs),
            "vendor_watchlisted": vendor.get("watchlisted"),
            "vendor_claims_serviced": vendor.get("claims_serviced"),
            "vendor_avg_invoice": vendor.get("avg_invoice_amount"),
            **graph_feats,
        }
        if doc_analysis.get("invoice_total"):
            ctx["estimated_repair_cost"] = ctx.get("estimated_repair_cost") or doc_analysis[
                "invoice_total"
            ]

        scorer = get_scorer()
        scored = scorer.score(ctx)
        prob = float(scored["fraud_probability"])

        data = self.ask(
            "anomaly_explanation",
            {"fraud_probability": prob, "top_features": scored["top_features"]},
            instruction=INSTRUCTION,
            schema=SCHEMA,
        )

        signals = []
        if prob >= 0.8:
            signals.append(
                signal(
                    self.name,
                    "MODEL_HIGH_RISK",
                    f"Anomaly model scored this claim at {prob:.0%} fraud probability.",
                    "high",
                    0.30,
                )
            )
        elif prob >= 0.5:
            signals.append(
                signal(
                    self.name,
                    "MODEL_ELEVATED_RISK",
                    f"Anomaly model scored this claim at {prob:.0%} fraud probability.",
                    "medium",
                    0.15,
                )
            )

        return AgentResult(
            name=self.name,
            output={
                "fraud_probability": prob,
                "model": scored["model"],
                "model_trained": scored["trained"],
                "top_features": scored["top_features"],
                **data,
            },
            summary=data.get("explanation", f"Fraud probability {prob:.0%}."),
            signals=signals,
            evidence=[
                {
                    "type": "model_feature",
                    "label": f["name"],
                    "detail": f,
                }
                for f in scored["top_features"][:5]
            ],
            risk_contribution=sum(s["weight"] for s in signals),
        )

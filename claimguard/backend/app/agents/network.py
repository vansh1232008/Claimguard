"""Agent 6 — Network / fraud-ring analysis.

Looks at the claim's neighbourhood in the relationship graph. A claim that is
clean on its own is a different proposition when it shares a bank account with
four other claims filed the same month through the same garage.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.agents.state import AgentResult, InvestigationState, signal
from app.services.graphstore import get_graph_store

SCHEMA: dict[str, Any] = {
    "ring_risk": "none|low|moderate|high",
    "ring_size": "integer",
    "shared_attributes": ["string"],
    "cohesion": "number between 0 and 1",
    "suspicious_links": [
        {
            "claim_number": "string",
            "relationship": "string",
            "shared_value": "string",
            "claimed_amount": "number",
        }
    ],
    "narrative": "string",
    "recommended_expansion": ["string"],
}

INSTRUCTION = (
    "Assess whether this claim belongs to an organised group. Weigh identity links "
    "(shared phone, shared bank account, shared address) far more heavily than "
    "incidental ones (same repair garage, same postcode). Say plainly if the links "
    "look coincidental."
)


class NetworkAgent(BaseAgent):
    name = "network"
    state_key = "network"
    description = "Graph-based fraud ring detection"

    def run(self, state: InvestigationState) -> AgentResult:
        claim_id = state["claim_id"]
        cached = state.get("_graph_context") or {}  # type: ignore[typeddict-item]
        store = get_graph_store()

        ring = cached.get("ring") or store.ring_for_claim(claim_id)
        neighbours = cached.get("neighbours") or store.neighbours(claim_id, depth=1)

        data = self.ask(
            "network_assessment",
            {"ring": ring, "neighbour_claims": neighbours[:10]},
            instruction=INSTRUCTION,
            schema=SCHEMA,
        )

        signals = []
        risk = str(data.get("ring_risk", "none"))
        identity_links = [
            n
            for n in neighbours
            if any(
                k in (n.get("relationship") or "")
                for k in ("SHARED_PHONE", "SHARED_BANK_ACCOUNT", "SHARED_ADDRESS")
            )
        ]
        if risk == "high":
            signals.append(
                signal(
                    self.name,
                    "FRAUD_RING_HIGH",
                    f"Claim sits in a tightly connected cluster of {ring.get('size', 0)} claims.",
                    "high",
                    0.30,
                )
            )
        elif risk == "moderate":
            signals.append(
                signal(
                    self.name,
                    "FRAUD_RING_MODERATE",
                    f"Claim is linked to {ring.get('size', 1) - 1} other claims through "
                    f"{', '.join(ring.get('shared_attributes', [])) or 'shared identifiers'}.",
                    "medium",
                    0.15,
                )
            )
        for link in identity_links[:3]:
            signals.append(
                signal(
                    self.name,
                    "SHARED_IDENTITY",
                    (
                        f"Shares {link['relationship'].replace('SHARED_', '').lower()} with claim "
                        f"{link.get('claim_number')} ({link.get('customer_name')})."
                    ),
                    "high",
                    0.12,
                )
            )

        evidence = [
            {
                "type": "graph_link",
                "label": f"{n.get('claim_number')} · {n.get('relationship')}",
                "detail": n,
            }
            for n in neighbours[:8]
        ]

        return AgentResult(
            name=self.name,
            output={**data, "ring": ring, "graph_backend": store.name},
            summary=data.get("narrative", "Network analysed."),
            signals=signals,
            evidence=evidence,
            risk_contribution=sum(s["weight"] for s in signals),
        )

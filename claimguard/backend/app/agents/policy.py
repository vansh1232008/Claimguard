"""Agent 4 — Policy coverage (RAG).

Retrieves the clauses that actually govern this loss from the policy wording
and decides coverage against them. Retrieval matters here for a specific
reason: a decline has to cite the clause it rests on, so the agent is built to
return clause IDs, not a summary of the policy from memory.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.agents.state import AgentResult, InvestigationState, signal
from app.config import settings
from app.services.vectorstore import get_vector_store

SCHEMA: dict[str, Any] = {
    "covered": "boolean",
    "coverage_confidence": "number between 0 and 1",
    "applicable_clauses": [{"clause_id": "string", "title": "string", "score": "number"}],
    "exclusions_triggered": [{"exclusion": "string", "matched_on": "string"}],
    "deductible_applied": "number",
    "coverage_limit": "number",
    "payable_estimate": "number",
    "reasoning": "string",
}

INSTRUCTION = (
    "Decide coverage using ONLY the retrieved policy clauses. State which clause "
    "grants or excludes cover, apply the deductible and the limit, and give the "
    "payable amount. If the clauses do not settle the question, say the coverage is "
    "uncertain rather than assuming cover."
)


class PolicyAgent(BaseAgent):
    name = "coverage"
    state_key = "coverage"
    description = "Retrieval-augmented policy coverage determination"

    def _build_query(self, claim: dict[str, Any], policy: dict[str, Any]) -> str:
        return (
            f"{claim.get('incident_type', 'collision')} claim under "
            f"{policy.get('product', 'motor comprehensive')} policy. "
            f"{claim.get('incident_description', '')} "
            f"Amount claimed {claim.get('claimed_amount')}. "
            "Which clauses grant cover, which exclusions apply, what deductible and "
            "limit are payable?"
        )

    def run(self, state: InvestigationState) -> AgentResult:
        claim = state.get("claim", {})
        policy = state.get("policy", {})

        store = get_vector_store()
        query = self._build_query(claim, policy)
        clauses = store.search(query, top_k=settings.rag_top_k)

        data = self.ask(
            "policy_coverage_review",
            {
                "claim": claim,
                "policy": policy,
                "retrieved_clauses": [
                    {
                        "clause_id": c.get("clause_id"),
                        "title": c.get("title"),
                        "text": c.get("text", "")[:1200],
                        "score": c.get("score"),
                    }
                    for c in clauses
                ],
            },
            instruction=INSTRUCTION,
            schema=SCHEMA,
        )

        signals = []
        if not data.get("covered", True):
            signals.append(
                signal(
                    self.name,
                    "NOT_COVERED",
                    str(data.get("reasoning", "The loss is not covered by the policy.")),
                    "high",
                    0.25,
                )
            )
        for exc in data.get("exclusions_triggered", []) or []:
            signals.append(
                signal(
                    self.name,
                    "EXCLUSION_TRIGGERED",
                    f"Exclusion '{exc.get('exclusion')}' matched on '{exc.get('matched_on')}'.",
                    "high",
                    0.15,
                )
            )
        if float(data.get("coverage_confidence", 1.0)) < 0.5:
            signals.append(
                signal(
                    self.name,
                    "COVERAGE_UNCERTAIN",
                    "Retrieved wording does not clearly settle whether this loss is covered.",
                    "low",
                    0.05,
                )
            )

        evidence = [
            {
                "type": "policy_clause",
                "label": f"{c.get('clause_id')} {c.get('title')}",
                "detail": {
                    "source": c.get("source"),
                    "score": c.get("score"),
                    "excerpt": (c.get("text") or "")[:400],
                },
            }
            for c in clauses
        ]

        return AgentResult(
            name=self.name,
            output={**data, "retrieval_backend": store.name, "clauses_retrieved": len(clauses)},
            summary=data.get("reasoning", "Coverage assessed."),
            signals=signals,
            evidence=evidence,
            risk_contribution=sum(s["weight"] for s in signals),
        )

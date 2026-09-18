"""Agent 1 — Intake.

Normalises the claim, checks it is complete enough to investigate, and raises
the cheap first-pass flags (late reporting, inception-window incidents,
suspiciously thin narratives) before any expensive agent runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.state import AgentResult, InvestigationState, signal

SCHEMA: dict[str, Any] = {
    "normalized_incident_type": "string",
    "severity_band": "minor|moderate|major|severe",
    "completeness_score": "number between 0 and 1",
    "missing_fields": ["string"],
    "narrative_summary": "string",
    "early_flags": ["string"],
}

INSTRUCTION = (
    "Normalise this first notice of loss. Judge whether the claim as filed contains "
    "enough information to adjudicate, list any field an adjudicator would have to "
    "chase, and flag anything that is unusual about HOW the claim was reported "
    "(timing, missing police report, thin description). Do not judge fraud yet."
)


def _days(a: Any, b: Any) -> float:
    def parse(v: Any) -> datetime | None:
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        try:
            d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    da, db = parse(a), parse(b)
    if not da or not db:
        return 0.0
    return (da - db).total_seconds() / 86400.0


class IntakeAgent(BaseAgent):
    name = "intake"
    state_key = "intake"
    description = "Normalises and validates the first notice of loss"
    critical = True

    def run(self, state: InvestigationState) -> AgentResult:
        claim = dict(state.get("claim", {}))
        policy = state.get("policy", {})
        customer = state.get("customer", {})

        reporting_delay = max(0.0, _days(claim.get("reported_date"), claim.get("incident_date")))
        since_start = max(0.0, _days(claim.get("incident_date"), policy.get("start_date")))
        claim["reporting_delay_days"] = round(reporting_delay, 2)
        claim["days_since_policy_start"] = round(since_start, 2)
        claim["customer_name"] = customer.get("full_name")

        data = self.ask(
            "intake_normalisation",
            {"claim": claim, "policy": policy, "customer": customer},
            instruction=INSTRUCTION,
            schema=SCHEMA,
        )

        signals = []
        if reporting_delay > 14:
            signals.append(
                signal(
                    self.name,
                    "LATE_REPORTING",
                    f"Claim reported {reporting_delay:.0f} days after the incident date.",
                    "medium",
                    0.10,
                )
            )
        if since_start < 30:
            signals.append(
                signal(
                    self.name,
                    "INCEPTION_WINDOW",
                    f"Incident occurred {since_start:.0f} days after the policy started.",
                    "high",
                    0.15,
                )
            )
        if float(data.get("completeness_score", 1.0)) < 0.6:
            signals.append(
                signal(
                    self.name,
                    "INCOMPLETE_FILING",
                    "Claim is missing information an adjudicator needs: "
                    + ", ".join(data.get("missing_fields", []) or ["unspecified fields"]),
                    "low",
                    0.05,
                )
            )
        if not claim.get("police_report") and float(claim.get("claimed_amount") or 0) > 10000:
            signals.append(
                signal(
                    self.name,
                    "NO_POLICE_REPORT",
                    "High-value claim filed without a police report.",
                    "medium",
                    0.08,
                )
            )

        # Keep the enriched claim in state for downstream agents.
        state["claim"] = claim

        summary = data.get("narrative_summary") or "Claim normalised."
        return AgentResult(
            name=self.name,
            output={
                **data,
                "reporting_delay_days": round(reporting_delay, 2),
                "days_since_policy_start": round(since_start, 2),
            },
            summary=summary,
            signals=signals,
            risk_contribution=sum(s["weight"] for s in signals),
        )

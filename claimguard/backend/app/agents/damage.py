"""Agent 3 — Damage analysis.

Runs the vision module over the claim photographs and asks: does the damage in
these pictures plausibly cost what the claimant says it costs? A severe-looking
number attached to a minor scuff is one of the most common padding patterns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agents.base import BaseAgent
from app.agents.state import AgentResult, InvestigationState, signal
from app.services.vision import analyse_images

SCHEMA: dict[str, Any] = {
    "damage_severity": "none_detected|minor|moderate|severe",
    "regions_detected": ["string"],
    "plausibility_score": "number between 0 and 1",
    "expected_cost_range": ["number", "number"],
    "inconsistencies": ["string"],
    "notes": "string",
}

INSTRUCTION = (
    "Compare the detected damage with the amount being claimed and the repair "
    "estimate. Decide whether the money asked for is plausible for the damage shown, "
    "and list any inconsistency (claimed regions that are not visible, amounts far "
    "above the range for that severity, no photographs at all)."
)


class DamageAgent(BaseAgent):
    name = "damage_analysis"
    state_key = "damage_analysis"
    description = "Vision-based damage assessment and cost plausibility check"

    def run(self, state: InvestigationState) -> AgentResult:
        claim = state.get("claim", {})
        options = state.get("options", {}) or {}
        paths = [Path(p) for p in (state.get("image_paths") or [])]

        if not options.get("include_damage_analysis", True):
            return AgentResult(
                name=self.name,
                output={"skipped": True},
                summary="Damage analysis skipped by request.",
                status="skipped",
            )

        vision = analyse_images(paths) if paths else {"backend": "none", "findings": []}
        findings = vision.get("findings", [])

        data = self.ask(
            "damage_assessment",
            {"claim": claim, "image_findings": findings, "vision_backend": vision.get("backend")},
            instruction=INSTRUCTION,
            schema=SCHEMA,
        )

        signals = []
        plausibility = float(data.get("plausibility_score", 1.0))
        if plausibility < 0.5:
            signals.append(
                signal(
                    self.name,
                    "DAMAGE_COST_MISMATCH",
                    "Claimed amount is not plausible for the damage observed in the photographs.",
                    "high",
                    0.18,
                )
            )
        elif plausibility < 0.8:
            signals.append(
                signal(
                    self.name,
                    "DAMAGE_COST_STRETCHED",
                    "Claimed amount sits above the usual range for this damage severity.",
                    "medium",
                    0.08,
                )
            )
        if not findings:
            signals.append(
                signal(
                    self.name,
                    "NO_DAMAGE_EVIDENCE",
                    "No damage photographs were supplied for this claim.",
                    "medium",
                    0.09,
                )
            )

        evidence = [
            {
                "type": "damage_region",
                "label": f"{f.get('region')} - {f.get('damage_type')} ({f.get('severity')})",
                "detail": f,
            }
            for f in findings
        ]

        return AgentResult(
            name=self.name,
            output={**data, "vision_backend": vision.get("backend"), "raw_findings": findings},
            summary=data.get("notes", "Damage assessed."),
            signals=signals,
            evidence=evidence,
            risk_contribution=sum(s["weight"] for s in signals),
        )

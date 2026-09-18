"""Agent 2 — Document analysis.

Reads the paperwork attached to the claim (invoices, estimates, police
reports), extracts the fields that matter, and cross-checks them against what
the claimant declared. Most padded claims fail here first: the invoice says one
number, the claim form says another.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agents.base import BaseAgent
from app.agents.state import AgentResult, InvestigationState, signal
from app.services.documents import process_document

SCHEMA: dict[str, Any] = {
    "consistency_score": "number between 0 and 1",
    "documents_reviewed": "integer",
    "mismatches": [
        {
            "field": "string",
            "document": "string",
            "claim_value": "string or number",
            "document_value": "string or number",
            "severity": "low|medium|high",
        }
    ],
    "duplicate_invoice_numbers": ["string"],
    "invoice_total": "number or null",
    "notes": "string",
}

INSTRUCTION = (
    "Cross-examine the supporting documents against the claim as filed. Report every "
    "field where the paperwork disagrees with the declared claim (amounts, dates, "
    "vehicle registration, vendor identity), and call out invoice numbers that repeat "
    "across claims. Only report a mismatch you can point to in the supplied data."
)


class DocumentAgent(BaseAgent):
    name = "document_analysis"
    state_key = "document_analysis"
    description = "Extracts and cross-checks invoices, estimates and reports"

    def run(self, state: InvestigationState) -> AgentResult:
        claim = state.get("claim", {})
        stored_docs = state.get("documents", []) or []

        # Extract fields for any document that has not been parsed yet.
        parsed: list[dict[str, Any]] = []
        for doc in stored_docs:
            fields = doc.get("extracted_fields") or {}
            if not fields and doc.get("storage_path"):
                path = Path(doc["storage_path"])
                if path.exists():
                    processed = process_document(path, doc.get("doc_type", "invoice"))
                    fields = processed["extracted_fields"]
                    doc["extracted_fields"] = fields
                    doc["raw_text"] = processed["raw_text"]
            parsed.append(
                {
                    "id": doc.get("id"),
                    "filename": doc.get("filename"),
                    "doc_type": doc.get("doc_type"),
                    "extracted_fields": fields,
                }
            )

        data = self.ask(
            "document_consistency",
            {"claim": claim, "documents": parsed},
            instruction=INSTRUCTION,
            schema=SCHEMA,
        )

        signals = []
        for mismatch in data.get("mismatches", []) or []:
            severity = str(mismatch.get("severity", "medium"))
            weight = {"high": 0.18, "medium": 0.10, "low": 0.04}.get(severity, 0.08)
            signals.append(
                signal(
                    self.name,
                    "DOCUMENT_MISMATCH",
                    (
                        f"{mismatch.get('field')} differs between the claim "
                        f"({mismatch.get('claim_value')}) and {mismatch.get('document')} "
                        f"({mismatch.get('document_value')})."
                    ),
                    severity,
                    weight,
                )
            )
        for dup in data.get("duplicate_invoice_numbers", []) or []:
            signals.append(
                signal(
                    self.name,
                    "DUPLICATE_INVOICE",
                    f"Invoice number {dup} appears more than once in the evidence pack.",
                    "high",
                    0.20,
                )
            )
        if not parsed:
            signals.append(
                signal(
                    self.name,
                    "NO_DOCUMENTS",
                    "No supporting documents were attached, so the amount is uncorroborated.",
                    "medium",
                    0.10,
                )
            )

        evidence = [
            {
                "type": "document",
                "label": d["filename"],
                "detail": d.get("extracted_fields", {}),
            }
            for d in parsed
        ]

        return AgentResult(
            name=self.name,
            output=data,
            summary=data.get("notes", "Documents reviewed."),
            signals=signals,
            evidence=evidence,
            risk_contribution=sum(s["weight"] for s in signals),
        )

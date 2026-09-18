"""Run an investigation and persist the whole audit trail."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.pipeline import get_pipeline
from app.agents.state import new_state
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.models import AgentRun, Claim, Customer, Investigation, Policy, Vehicle, Vendor

logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_context(db: Session, claim: Claim) -> dict[str, Any]:
    """Assemble everything the agents need, in plain dicts."""
    customer = db.get(Customer, claim.customer_id)
    policy = db.get(Policy, claim.policy_id)
    vehicle = db.get(Vehicle, claim.vehicle_id) if claim.vehicle_id else None
    vendor = db.get(Vendor, claim.vendor_id) if claim.vendor_id else None

    documents = [
        {
            "id": d.id,
            "doc_type": d.doc_type,
            "filename": d.filename,
            "storage_path": d.storage_path,
            "extracted_fields": d.extracted_fields or {},
            "raw_text": d.raw_text,
        }
        for d in claim.documents
    ]
    image_paths = [
        d["storage_path"]
        for d in documents
        if d.get("storage_path")
        and str(d["storage_path"]).lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    within_period = True
    if policy and policy.end_date and claim.incident_date:
        start = policy.start_date
        end = policy.end_date
        inc = claim.incident_date
        try:
            within_period = bool(start <= inc <= end)
        except TypeError:
            within_period = True

    return {
        "claim": {
            "id": claim.id,
            "claim_number": claim.claim_number,
            "incident_type": claim.incident_type,
            "incident_date": claim.incident_date,
            "reported_date": claim.reported_date,
            "incident_description": claim.incident_description,
            "incident_postal_code": claim.incident_postal_code,
            "claimed_amount": claim.claimed_amount,
            "estimated_repair_cost": claim.estimated_repair_cost,
            "witnesses": claim.witnesses,
            "police_report": claim.police_report,
            "injury_claimed": claim.injury_claimed,
            "prior_claims_12m": claim.prior_claims_12m,
            "within_policy_period": within_period,
            "vehicle_registration": vehicle.vin if vehicle else None,
        },
        "policy": {
            "policy_number": policy.policy_number if policy else None,
            "product": policy.product if policy else "motor_comprehensive",
            "status": policy.status if policy else "unknown",
            "start_date": policy.start_date if policy else None,
            "end_date": policy.end_date if policy else None,
            "annual_premium": policy.annual_premium if policy else 0.0,
            "coverage_limit": policy.coverage_limit if policy else 25000.0,
            "deductible": policy.deductible if policy else 500.0,
            "exclusions": policy.exclusions if policy else [],
            "bonus_malus": policy.bonus_malus if policy else 50,
        },
        "customer": {
            "id": customer.id if customer else None,
            "full_name": customer.full_name if customer else None,
            "tenure_months": customer.tenure_months if customer else 0,
            "postal_code": customer.postal_code if customer else None,
            "region": customer.region if customer else None,
        },
        "vehicle": {
            "vin": vehicle.vin if vehicle else None,
            "make": vehicle.make if vehicle else None,
            "model": vehicle.model if vehicle else None,
            "year": vehicle.year if vehicle else None,
            "age": (_now().year - vehicle.year) if vehicle else 0,
            "vehicle_power": vehicle.vehicle_power if vehicle else 0,
            "declared_value": vehicle.declared_value if vehicle else 0.0,
        },
        "vendor": {
            "id": vendor.id if vendor else None,
            "name": vendor.name if vendor else None,
            "vendor_type": vendor.vendor_type if vendor else None,
            "watchlisted": vendor.watchlisted if vendor else False,
            "claims_serviced": vendor.claims_serviced if vendor else 0,
            "avg_invoice_amount": vendor.avg_invoice_amount if vendor else 0.0,
        },
        "documents": documents,
        "image_paths": image_paths,
    }


def investigate_claim(
    db: Session, claim_id: str, options: dict[str, Any] | None = None
) -> Investigation:
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise NotFoundError(f"Claim {claim_id} not found")

    options = options or {}
    investigation = Investigation(claim_id=claim.id, status="running", started_at=_now())
    db.add(investigation)
    db.flush()

    context = build_context(db, claim)
    state = new_state(claim.id, options=options, **context)

    pipeline = get_pipeline()
    try:
        final = pipeline.run(state)
    except Exception as exc:  # pragma: no cover - pipeline itself guards agents
        logger.exception("Investigation failed for claim %s", claim.claim_number)
        investigation.status = "failed"
        investigation.error = str(exc)
        investigation.finished_at = _now()
        db.commit()
        return investigation

    adjudication = final.get("adjudication", {}) or {}
    investigation.status = "completed"
    investigation.decision = adjudication.get("decision", "review")
    investigation.fraud_score = float(final.get("fraud_score", 0.0))
    investigation.confidence = float(adjudication.get("confidence", 0.5))
    investigation.rationale = adjudication.get("rationale")
    investigation.recommended_payout = float(adjudication.get("recommended_payout") or 0.0)
    investigation.risk_signals = final.get("risk_signals", [])
    investigation.evidence = final.get("evidence", [])
    investigation.llm_calls = int(final.get("llm_calls", 0))
    investigation.duration_ms = int(final.get("duration_ms", 0))
    investigation.finished_at = _now()
    investigation.agent_summary = {
        "engine": final.get("engine"),
        "agents_run": [r["agent_name"] for r in final.get("agent_runs", [])],
        "errors": final.get("errors", []),
        "coverage": final.get("coverage", {}).get("covered"),
        "model": (final.get("anomaly", {}) or {}).get("model"),
        "ring_size": ((final.get("network", {}) or {}).get("ring") or {}).get("size", 0),
    }

    for i, run in enumerate(final.get("agent_runs", [])):
        db.add(
            AgentRun(
                investigation_id=investigation.id,
                agent_name=run["agent_name"],
                sequence=i,
                status=run.get("status", "ok"),
                summary=run.get("summary"),
                output=run.get("output") or {},
                risk_contribution=float(run.get("risk_contribution") or 0.0),
                llm_calls=int(run.get("llm_calls") or 0),
                duration_ms=int(run.get("duration_ms") or 0),
                error=run.get("error"),
            )
        )

    claim.status = {
        "approve": "approved",
        "review": "under_review",
        "reject": "rejected",
    }.get(investigation.decision or "review", "under_review")

    db.commit()
    db.refresh(investigation)
    logger.info(
        "Claim %s -> %s (score %.2f, %d agents, %d LLM calls, %d ms)",
        claim.claim_number,
        investigation.decision,
        investigation.fraud_score or 0.0,
        len(final.get("agent_runs", [])),
        investigation.llm_calls,
        investigation.duration_ms,
    )
    return investigation


def latest_investigation(db: Session, claim_id: str) -> Investigation | None:
    return db.execute(
        select(Investigation)
        .where(Investigation.claim_id == claim_id)
        .order_by(Investigation.started_at.desc())
        .limit(1)
    ).scalar_one_or_none()

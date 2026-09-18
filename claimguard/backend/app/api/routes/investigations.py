"""Investigation endpoints: run the pipeline, read the audit trail."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.agents.pipeline import AGENT_ORDER, get_pipeline
from app.core.exceptions import NotFoundError
from app.database import get_db
from app.models import Claim, Investigation
from app.schemas import InvestigationOut, InvestigationRequest
from app.services.investigation import investigate_claim, latest_investigation

router = APIRouter(tags=["investigations"])


@router.post("/claims/{claim_id}/investigate", response_model=InvestigationOut)
def run_investigation(
    claim_id: str, payload: InvestigationRequest | None = None, db: Session = Depends(get_db)
):
    payload = payload or InvestigationRequest()
    if not payload.force:
        existing = latest_investigation(db, claim_id)
        if existing and existing.status == "completed":
            return existing
    return investigate_claim(
        db,
        claim_id,
        options={
            "include_damage_analysis": payload.include_damage_analysis,
            "notes": payload.notes,
        },
    )


@router.get("/claims/{claim_id}/investigations", response_model=list[InvestigationOut])
def claim_investigations(claim_id: str, db: Session = Depends(get_db)):
    if db.get(Claim, claim_id) is None:
        raise NotFoundError(f"Claim {claim_id} not found")
    return (
        db.execute(
            select(Investigation)
            .options(selectinload(Investigation.agent_runs))
            .where(Investigation.claim_id == claim_id)
            .order_by(Investigation.started_at.desc())
        )
        .scalars()
        .all()
    )


@router.get("/investigations", response_model=list[InvestigationOut])
def list_investigations(
    db: Session = Depends(get_db),
    decision: str | None = None,
    min_score: float | None = None,
    limit: int = Query(50, le=500),
):
    stmt = (
        select(Investigation)
        .options(selectinload(Investigation.agent_runs))
        .order_by(Investigation.started_at.desc())
    )
    if decision:
        stmt = stmt.where(Investigation.decision == decision)
    if min_score is not None:
        stmt = stmt.where(Investigation.fraud_score >= min_score)
    return db.execute(stmt.limit(limit)).scalars().all()


@router.get("/investigations/{investigation_id}", response_model=InvestigationOut)
def get_investigation(investigation_id: str, db: Session = Depends(get_db)):
    inv = db.execute(
        select(Investigation)
        .options(selectinload(Investigation.agent_runs))
        .where(Investigation.id == investigation_id)
    ).scalar_one_or_none()
    if inv is None:
        raise NotFoundError(f"Investigation {investigation_id} not found")
    return inv


@router.post("/investigations/batch")
def batch_investigate(
    db: Session = Depends(get_db),
    limit: int = Query(10, le=200, description="How many un-investigated claims to process"),
):
    """Process a queue of new claims. Handy for demos and for benchmarking."""
    pending = (
        db.execute(
            select(Claim).where(Claim.status == "submitted").order_by(Claim.created_at.asc()).limit(limit)
        )
        .scalars()
        .all()
    )
    results = []
    for claim in pending:
        inv = investigate_claim(db, claim.id)
        results.append(
            {
                "claim_number": claim.claim_number,
                "decision": inv.decision,
                "fraud_score": inv.fraud_score,
                "duration_ms": inv.duration_ms,
            }
        )
    return {"processed": len(results), "results": results}


@router.get("/agents")
def describe_agents():
    """What the pipeline is made of - used by the frontend's timeline view."""
    pipeline = get_pipeline()
    return {
        "engine": pipeline.engine,
        "order": AGENT_ORDER,
        "agents": [
            {
                "name": name,
                "description": agent.description,
                "critical": agent.critical,
            }
            for name, agent in pipeline.agents.items()
        ],
    }

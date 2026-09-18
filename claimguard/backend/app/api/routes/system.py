"""Health, configuration visibility and stats."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.pipeline import get_pipeline
from app.config import settings
from app.database import get_db
from app.models import Claim, Investigation
from app.schemas import StatsOut
from app.services.graphstore import get_graph_store
from app.services.llm import get_llm
from app.services.ml import get_scorer
from app.services.vectorstore import get_vector_store, reindex_policies

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@router.get("/system/info")
def system_info():
    """Which backend every swappable subsystem is currently running on.

    The frontend shows this as a row of chips so it is always obvious whether
    you are looking at offline output or the real stack.
    """
    scorer = get_scorer()
    store = get_vector_store()
    return {
        "llm": {"mode": get_llm().mode, "model": settings.gemini_model},
        "vector_store": {"mode": store.name, "clauses": store.count()},
        "graph_store": {"mode": get_graph_store().name},
        "fraud_model": {"mode": scorer.mode, "features": len(scorer.feature_names)},
        "pipeline": {"engine": get_pipeline().engine},
        "database": settings.database_url.split("://")[0],
        "thresholds": {
            "auto_approve_below": settings.auto_approve_below,
            "auto_reject_above": settings.auto_reject_above,
            "high_value_claim_threshold": settings.high_value_claim_threshold,
        },
    }


@router.post("/system/reindex-policies")
def reindex():
    return reindex_policies()


@router.post("/system/reload-models")
def reload_models():
    get_scorer().reload()
    return {"fraud_model": get_scorer().mode}


@router.get("/stats", response_model=StatsOut)
def stats(db: Session = Depends(get_db)):
    total_claims = db.execute(select(func.count(Claim.id))).scalar_one()
    total_exposure = db.execute(select(func.sum(Claim.claimed_amount))).scalar_one() or 0.0

    investigations = (
        db.execute(select(Investigation).where(Investigation.status == "completed"))
        .scalars()
        .all()
    )
    decisions = Counter(i.decision for i in investigations)
    scores = [i.fraud_score or 0.0 for i in investigations]
    flagged = [i for i in investigations if (i.fraud_score or 0) >= settings.auto_approve_below]

    claim_amounts = {
        c.id: float(c.claimed_amount or 0.0)
        for c in db.execute(select(Claim)).scalars().all()
    }
    flagged_exposure = sum(claim_amounts.get(i.claim_id, 0.0) for i in flagged)

    by_day: dict[str, Counter] = {}
    for inv in investigations:
        day = inv.started_at.date().isoformat() if inv.started_at else "unknown"
        by_day.setdefault(day, Counter())[inv.decision or "review"] += 1

    buckets = [0] * 10
    for s in scores:
        buckets[min(9, int(s * 10))] += 1

    store = get_graph_store()
    rings = len(store.list_rings()) if store.graph.number_of_nodes() else 0

    return StatsOut(
        total_claims=total_claims,
        investigated=len(investigations),
        pending=max(0, total_claims - len(investigations)),
        approved=decisions.get("approve", 0),
        review=decisions.get("review", 0),
        rejected=decisions.get("reject", 0),
        avg_fraud_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
        total_exposure=round(float(total_exposure), 2),
        flagged_exposure=round(flagged_exposure, 2),
        rings_detected=rings,
        avg_duration_ms=round(
            sum(i.duration_ms for i in investigations) / len(investigations), 1
        )
        if investigations
        else 0.0,
        avg_llm_calls=round(sum(i.llm_calls for i in investigations) / len(investigations), 2)
        if investigations
        else 0.0,
        decisions_by_day=[
            {"date": day, **dict(counter)} for day, counter in sorted(by_day.items())
        ],
        score_histogram=[
            {"bucket": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "count": buckets[i]} for i in range(10)
        ],
    )

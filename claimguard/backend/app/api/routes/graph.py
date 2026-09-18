"""Graph endpoints: fraud rings and the claim neighbourhood explorer."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.database import get_db
from app.models import Claim
from app.schemas import FraudRingOut
from app.services.graph_sync import refresh_graph
from app.services.graphstore import get_graph_store

router = APIRouter(prefix="/graph", tags=["graph"])


@router.post("/refresh")
def rebuild_graph(db: Session = Depends(get_db), limit: int | None = None):
    return refresh_graph(db, limit=limit)


@router.get("/stats")
def graph_stats(db: Session = Depends(get_db)):
    store = get_graph_store()
    if store.graph.number_of_nodes() == 0:
        refresh_graph(db)
    return store.stats()


@router.get("/rings", response_model=list[FraudRingOut])
def list_rings(
    db: Session = Depends(get_db),
    min_size: int = Query(3, ge=2, le=50),
    limit: int = Query(25, le=200),
):
    store = get_graph_store()
    if store.graph.number_of_nodes() == 0:
        refresh_graph(db)
    rings = store.list_rings(min_size=min_size)[:limit]
    return [
        FraudRingOut(
            ring_id=r["ring_id"],
            size=r["size"],
            total_exposure=r["total_exposure"],
            shared_attributes=r["shared_attributes"],
            cohesion=r["cohesion"],
            members=[
                {
                    "claim_id": m["claim_id"],
                    "claim_number": m["claim_number"] or "",
                    "customer_name": m["customer_name"] or "",
                    "claimed_amount": m["claimed_amount"] or 0.0,
                    "fraud_score": m.get("fraud_score"),
                }
                for m in r["members"]
            ],
        )
        for r in rings
    ]


@router.get("/claims/{claim_id}")
def claim_subgraph(claim_id: str, depth: int = Query(2, ge=1, le=3), db: Session = Depends(get_db)):
    if db.get(Claim, claim_id) is None:
        raise NotFoundError(f"Claim {claim_id} not found")
    store = get_graph_store()
    if store.graph.number_of_nodes() == 0:
        refresh_graph(db)
    payload = store.subgraph_payload(claim_id, depth=depth)
    payload["ring"] = store.ring_for_claim(claim_id)
    return payload

"""Project the relational store into the relationship graph."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models import Claim, Customer, Investigation, Vendor
from app.services.graphstore import GraphClaim, get_graph_store

logger = get_logger(__name__)


def _week_key(dt: Any) -> str | None:
    if dt is None:
        return None
    try:
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except Exception:
        return None


def collect_graph_claims(db: Session, limit: int | None = None) -> list[GraphClaim]:
    stmt = (
        select(Claim, Customer, Vendor)
        .join(Customer, Claim.customer_id == Customer.id)
        .join(Vendor, Claim.vendor_id == Vendor.id, isouter=True)
        .order_by(Claim.created_at.desc())
    )
    if limit:
        stmt = stmt.limit(limit)

    # Latest fraud score per claim, so the graph view can colour nodes by risk.
    scores: dict[str, float] = {}
    for inv in db.execute(
        select(Investigation).order_by(Investigation.started_at.asc())
    ).scalars():
        if inv.fraud_score is not None:
            scores[inv.claim_id] = float(inv.fraud_score)

    rows: list[GraphClaim] = []
    for claim, customer, vendor in db.execute(stmt).all():
        rows.append(
            GraphClaim(
                claim_id=claim.id,
                claim_number=claim.claim_number,
                customer_id=customer.id,
                customer_name=customer.full_name,
                phone=customer.phone,
                bank_account_hash=customer.bank_account_hash,
                address=customer.address,
                vehicle_id=claim.vehicle_id,
                vendor_id=claim.vendor_id,
                vendor_name=vendor.name if vendor else None,
                postal_code=claim.incident_postal_code or customer.postal_code,
                incident_week=_week_key(claim.incident_date),
                claimed_amount=float(claim.claimed_amount or 0.0),
                fraud_score=scores.get(claim.id),
            )
        )
    return rows


def refresh_graph(db: Session, limit: int | None = None) -> dict[str, Any]:
    store = get_graph_store()
    claims = collect_graph_claims(db, limit=limit)
    result = store.rebuild(claims)
    logger.info(
        "Graph rebuilt: %s nodes / %s edges (%s)",
        result.get("nodes"),
        result.get("edges"),
        result.get("backend"),
    )
    return result


def graph_features_for_claim(claim_id: str) -> dict[str, Any]:
    """The graph-derived slice of the model's feature vector."""
    store = get_graph_store()
    ring = store.ring_for_claim(claim_id)
    neighbours = store.neighbours(claim_id, depth=1)

    shared_phone = sum(1 for n in neighbours if "SHARED_PHONE" in (n["relationship"] or ""))
    shared_account = sum(
        1 for n in neighbours if "SHARED_BANK_ACCOUNT" in (n["relationship"] or "")
    )
    vendor_degree = sum(1 for n in neighbours if "SHARED_VENDOR" in (n["relationship"] or ""))
    amounts = [n["claimed_amount"] for n in neighbours if n.get("claimed_amount")]
    return {
        "ring_size": ring.get("size", 0),
        "ring_cohesion": ring.get("cohesion", 0.0),
        "shared_phone_degree": shared_phone,
        "shared_account_degree": shared_account,
        "vendor_degree": vendor_degree,
        "neighbour_mean_amount": (sum(amounts) / len(amounts)) if amounts else 0.0,
        "_ring": ring,
        "_neighbours": neighbours,
    }

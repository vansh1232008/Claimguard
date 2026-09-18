"""Claim endpoints: list, detail, create, upload documents."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.database import get_db
from app.models import Claim, ClaimDocument, Customer, Policy, Vendor
from app.schemas import ClaimCreate, ClaimDetail, ClaimOut, DocumentOut
from app.services.documents import process_document
from app.services.investigation import latest_investigation

router = APIRouter(prefix="/claims", tags=["claims"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


@router.get("", response_model=list[ClaimOut])
def list_claims(
    db: Session = Depends(get_db),
    status: str | None = Query(None, description="submitted|under_review|approved|rejected"),
    decision: str | None = Query(None, description="approve|review|reject"),
    min_amount: float | None = None,
    search: str | None = Query(None, description="claim number or customer name"),
    limit: int = Query(50, le=500),
    offset: int = 0,
):
    stmt = select(Claim).order_by(Claim.created_at.desc())
    if status:
        stmt = stmt.where(Claim.status == status)
    if min_amount is not None:
        stmt = stmt.where(Claim.claimed_amount >= min_amount)
    if search:
        like = f"%{search.lower()}%"
        stmt = stmt.join(Customer, Claim.customer_id == Customer.id).where(
            func.lower(Claim.claim_number).like(like) | func.lower(Customer.full_name).like(like)
        )
    claims = db.execute(stmt.offset(offset).limit(limit)).scalars().all()

    if decision:
        keep = []
        for c in claims:
            inv = latest_investigation(db, c.id)
            if inv and inv.decision == decision:
                keep.append(c)
        claims = keep
    return claims


@router.get("/{claim_id}", response_model=ClaimDetail)
def get_claim(claim_id: str, db: Session = Depends(get_db)):
    claim = db.execute(
        select(Claim).options(selectinload(Claim.documents)).where(Claim.id == claim_id)
    ).scalar_one_or_none()
    if claim is None:
        raise NotFoundError(f"Claim {claim_id} not found")

    customer = db.get(Customer, claim.customer_id)
    policy = db.get(Policy, claim.policy_id)
    vendor = db.get(Vendor, claim.vendor_id) if claim.vendor_id else None
    inv = latest_investigation(db, claim.id)

    detail = ClaimDetail.model_validate(claim)
    detail.customer_name = customer.full_name if customer else None
    detail.policy_number = policy.policy_number if policy else None
    detail.vendor_name = vendor.name if vendor else None
    if inv:
        from app.schemas import InvestigationOut

        detail.latest_investigation = InvestigationOut.model_validate(inv)
    return detail


@router.post("", response_model=ClaimOut, status_code=201)
def create_claim(payload: ClaimCreate, db: Session = Depends(get_db)):
    policy = db.execute(
        select(Policy).where(Policy.policy_number == payload.policy_number)
    ).scalar_one_or_none()
    if policy is None:
        raise ValidationError(f"Unknown policy {payload.policy_number}")

    vendor = None
    if payload.vendor_external_id:
        vendor = db.execute(
            select(Vendor).where(Vendor.external_id == payload.vendor_external_id)
        ).scalar_one_or_none()

    prior = db.execute(
        select(func.count(Claim.id)).where(Claim.customer_id == policy.customer_id)
    ).scalar_one()

    claim = Claim(
        claim_number=f"CLM-{uuid.uuid4().hex[:10].upper()}",
        customer_id=policy.customer_id,
        policy_id=policy.id,
        vehicle_id=policy.vehicle_id,
        vendor_id=vendor.id if vendor else None,
        incident_type=payload.incident_type,
        incident_date=payload.incident_date,
        reported_date=_now(),
        incident_description=payload.incident_description,
        incident_postal_code=payload.incident_postal_code,
        claimed_amount=payload.claimed_amount,
        estimated_repair_cost=payload.estimated_repair_cost,
        witnesses=payload.witnesses,
        police_report=payload.police_report,
        injury_claimed=payload.injury_claimed,
        prior_claims_12m=int(prior),
        status="submitted",
    )
    db.add(claim)
    db.commit()
    db.refresh(claim)
    return claim


@router.post("/{claim_id}/documents", response_model=list[DocumentOut], status_code=201)
async def upload_documents(
    claim_id: str,
    files: list[UploadFile] = File(...),
    doc_type: str = Query("invoice"),
    db: Session = Depends(get_db),
):
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise NotFoundError(f"Claim {claim_id} not found")

    created: list[ClaimDocument] = []
    for upload in files:
        safe_name = (upload.filename or "document").replace("/", "_")
        dest = settings.uploads / f"{claim.claim_number}_{uuid.uuid4().hex[:6]}_{safe_name}"
        dest.write_bytes(await upload.read())

        processed = process_document(dest, doc_type)
        doc = ClaimDocument(
            claim_id=claim.id,
            doc_type=doc_type,
            filename=safe_name,
            storage_path=str(dest),
            mime_type=upload.content_type or "application/octet-stream",
            raw_text=processed["raw_text"],
            extracted_fields=processed["extracted_fields"],
        )
        db.add(doc)
        created.append(doc)

    db.commit()
    for doc in created:
        db.refresh(doc)
    return created


@router.get("/{claim_id}/documents", response_model=list[DocumentOut])
def list_documents(claim_id: str, db: Session = Depends(get_db)):
    claim = db.execute(
        select(Claim).options(selectinload(Claim.documents)).where(Claim.id == claim_id)
    ).scalar_one_or_none()
    if claim is None:
        raise NotFoundError(f"Claim {claim_id} not found")
    return claim.documents

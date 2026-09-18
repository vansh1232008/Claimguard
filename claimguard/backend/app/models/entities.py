"""Relational schema for ClaimGuard.

The graph database holds *relationships* between these entities for ring
detection; this relational store holds the record of truth for the claim, the
investigation and every agent step (the audit trail an insurer needs).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160))
    email: Mapped[str | None] = mapped_column(String(160), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)
    address: Mapped[str | None] = mapped_column(String(255), nullable=True)
    postal_code: Mapped[str | None] = mapped_column(String(16), index=True, nullable=True)
    region: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bank_account_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    tenure_months: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    policies: Mapped[list["Policy"]] = relationship(back_populates="customer")
    claims: Mapped[list["Claim"]] = relationship(back_populates="customer")


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    vin: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    make: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(64))
    year: Mapped[int] = mapped_column(Integer)
    vehicle_power: Mapped[int] = mapped_column(Integer, default=6)
    vehicle_gas: Mapped[str] = mapped_column(String(16), default="Regular")
    declared_value: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Vendor(Base):
    """Repair garages, clinics and assessors that appear on invoices."""

    __tablename__ = "vendors"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(160))
    vendor_type: Mapped[str] = mapped_column(String(32), default="garage")
    postal_code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    watchlisted: Mapped[bool] = mapped_column(Boolean, default=False)
    avg_invoice_amount: Mapped[float] = mapped_column(Float, default=0.0)
    claims_serviced: Mapped[int] = mapped_column(Integer, default=0)


class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    policy_number: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    vehicle_id: Mapped[str | None] = mapped_column(ForeignKey("vehicles.id"), nullable=True)
    product: Mapped[str] = mapped_column(String(64), default="motor_comprehensive")
    status: Mapped[str] = mapped_column(String(24), default="active")
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    annual_premium: Mapped[float] = mapped_column(Float, default=0.0)
    coverage_limit: Mapped[float] = mapped_column(Float, default=25000.0)
    deductible: Mapped[float] = mapped_column(Float, default=500.0)
    exclusions: Mapped[list | None] = mapped_column(JSON, default=list)
    bonus_malus: Mapped[int] = mapped_column(Integer, default=50)

    customer: Mapped[Customer] = relationship(back_populates="policies")


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    claim_number: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    policy_id: Mapped[str] = mapped_column(ForeignKey("policies.id"), index=True)
    vehicle_id: Mapped[str | None] = mapped_column(ForeignKey("vehicles.id"), nullable=True)
    vendor_id: Mapped[str | None] = mapped_column(ForeignKey("vendors.id"), index=True, nullable=True)

    incident_type: Mapped[str] = mapped_column(String(48), default="collision")
    incident_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    reported_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    incident_description: Mapped[str] = mapped_column(Text, default="")
    incident_postal_code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    claimed_amount: Mapped[float] = mapped_column(Float, default=0.0)
    estimated_repair_cost: Mapped[float] = mapped_column(Float, default=0.0)
    witnesses: Mapped[int] = mapped_column(Integer, default=0)
    police_report: Mapped[bool] = mapped_column(Boolean, default=False)
    injury_claimed: Mapped[bool] = mapped_column(Boolean, default=False)
    prior_claims_12m: Mapped[int] = mapped_column(Integer, default=0)

    status: Mapped[str] = mapped_column(String(24), default="submitted", index=True)
    # Ground truth is only present in the synthetic benchmark data.
    is_fraud_label: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    fraud_ring_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)

    extra: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    customer: Mapped[Customer] = relationship(back_populates="claims")
    documents: Mapped[list["ClaimDocument"]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )
    investigations: Mapped[list["Investigation"]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )


Index("ix_claims_status_created", Claim.status, Claim.created_at)


class ClaimDocument(Base):
    __tablename__ = "claim_documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id"), index=True)
    doc_type: Mapped[str] = mapped_column(String(32), default="invoice")
    filename: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(64), default="application/octet-stream")
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_fields: Mapped[dict | None] = mapped_column(JSON, default=dict)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    claim: Mapped[Claim] = relationship(back_populates="documents")


class Investigation(Base):
    """One end-to-end run of the 7-agent pipeline over one claim."""

    __tablename__ = "investigations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id"), index=True)
    status: Mapped[str] = mapped_column(String(24), default="running", index=True)

    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)  # approve/review/reject
    fraud_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommended_payout: Mapped[float | None] = mapped_column(Float, nullable=True)

    risk_signals: Mapped[list | None] = mapped_column(JSON, default=list)
    evidence: Mapped[list | None] = mapped_column(JSON, default=list)
    agent_summary: Mapped[dict | None] = mapped_column(JSON, default=dict)

    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    claim: Mapped[Claim] = relationship(back_populates="investigations")
    agent_runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="investigation", cascade="all, delete-orphan"
    )


class AgentRun(Base):
    """Audit record for a single agent inside one investigation."""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    investigation_id: Mapped[str] = mapped_column(ForeignKey("investigations.id"), index=True)
    agent_name: Mapped[str] = mapped_column(String(48), index=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output: Mapped[dict | None] = mapped_column(JSON, default=dict)
    risk_contribution: Mapped[float] = mapped_column(Float, default=0.0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    investigation: Mapped[Investigation] = relationship(back_populates="agent_runs")

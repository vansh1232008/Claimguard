"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- in
class ClaimCreate(BaseModel):
    policy_number: str = Field(..., description="Policy the claim is filed against")
    incident_type: str = "collision"
    incident_date: datetime
    incident_description: str = ""
    incident_postal_code: str | None = None
    claimed_amount: float = 0.0
    estimated_repair_cost: float = 0.0
    vendor_external_id: str | None = None
    witnesses: int = 0
    police_report: bool = False
    injury_claimed: bool = False


class InvestigationRequest(BaseModel):
    force: bool = Field(False, description="Re-run even if a finished investigation exists")
    include_damage_analysis: bool = True
    notes: str | None = None


# -------------------------------------------------------------------------- out
class DocumentOut(ORMModel):
    id: str
    doc_type: str
    filename: str
    mime_type: str
    extracted_fields: dict[str, Any] | None = None
    uploaded_at: datetime


class AgentRunOut(ORMModel):
    id: str
    agent_name: str
    sequence: int
    status: str
    summary: str | None
    output: dict[str, Any] | None
    risk_contribution: float
    llm_calls: int
    duration_ms: int
    error: str | None


class InvestigationOut(ORMModel):
    id: str
    claim_id: str
    status: str
    decision: str | None
    fraud_score: float | None
    confidence: float | None
    rationale: str | None
    recommended_payout: float | None
    risk_signals: list[Any] | None
    evidence: list[Any] | None
    agent_summary: dict[str, Any] | None
    llm_calls: int
    duration_ms: int
    error: str | None
    started_at: datetime
    finished_at: datetime | None
    agent_runs: list[AgentRunOut] = []


class ClaimOut(ORMModel):
    id: str
    claim_number: str
    customer_id: str
    policy_id: str
    incident_type: str
    incident_date: datetime
    reported_date: datetime
    claimed_amount: float
    estimated_repair_cost: float
    status: str
    is_fraud_label: bool | None
    fraud_ring_id: str | None
    created_at: datetime


class ClaimDetail(ClaimOut):
    incident_description: str
    incident_postal_code: str | None
    witnesses: int
    police_report: bool
    injury_claimed: bool
    prior_claims_12m: int
    customer_name: str | None = None
    policy_number: str | None = None
    vendor_name: str | None = None
    documents: list[DocumentOut] = []
    latest_investigation: InvestigationOut | None = None


class FraudRingMember(BaseModel):
    claim_id: str
    claim_number: str
    customer_name: str
    claimed_amount: float
    fraud_score: float | None = None


class FraudRingOut(BaseModel):
    ring_id: str
    size: int
    total_exposure: float
    shared_attributes: list[str]
    cohesion: float
    members: list[FraudRingMember]


class StatsOut(BaseModel):
    total_claims: int
    investigated: int
    pending: int
    approved: int
    review: int
    rejected: int
    avg_fraud_score: float
    total_exposure: float
    flagged_exposure: float
    rings_detected: int
    avg_duration_ms: float
    avg_llm_calls: float
    decisions_by_day: list[dict[str, Any]] = []
    score_histogram: list[dict[str, Any]] = []

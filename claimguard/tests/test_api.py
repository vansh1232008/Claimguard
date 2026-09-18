"""API-level tests. These exercise the real pipeline against a temp database."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

# Point at a throwaway database BEFORE app modules read the settings.
_TMP_DB = Path(tempfile.gettempdir()) / "claimguard_test.db"
_TMP_DB.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"

from fastapi.testclient import TestClient  # noqa: E402

from app.database import init_db, session_scope  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Claim, Customer, Policy, Vehicle, Vendor  # noqa: E402


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture(scope="module")
def client():
    init_db()
    with session_scope() as db:
        customer = Customer(
            external_id="CUS-TEST-1",
            full_name="Test Claimant",
            phone="+33600000001",
            address="1 Test Street, 75001 Paris",
            postal_code="75001",
            bank_account_hash="acct-test",
            tenure_months=48,
        )
        vehicle = Vehicle(
            vin="VFTEST0000001", make="Renault", model="Clio", year=2019, declared_value=12000
        )
        vendor = Vendor(external_id="VND-TEST", name="Test Garage", avg_invoice_amount=2000)
        db.add_all([customer, vehicle, vendor])
        db.flush()
        policy = Policy(
            policy_number="POL-TEST-1",
            customer_id=customer.id,
            vehicle_id=vehicle.id,
            start_date=_now() - timedelta(days=500),
            end_date=_now() + timedelta(days=200),
            annual_premium=700,
            coverage_limit=20000,
            deductible=500,
            exclusions=["racing", "driving_under_influence"],
        )
        db.add(policy)
        db.flush()
        db.add(
            Claim(
                claim_number="CLM-TEST-0001",
                customer_id=customer.id,
                policy_id=policy.id,
                vehicle_id=vehicle.id,
                vendor_id=vendor.id,
                incident_type="collision",
                incident_date=_now() - timedelta(days=12),
                reported_date=_now() - timedelta(days=11),
                incident_description=(
                    "Rear-ended at a red light on Rue de la Paix; the other driver "
                    "admitted fault and we exchanged details."
                ),
                incident_postal_code="75001",
                claimed_amount=4200,
                estimated_repair_cost=4000,
                witnesses=2,
                police_report=True,
            )
        )
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_system_info_reports_every_subsystem(client):
    info = client.get("/api/system/info").json()
    for key in ("llm", "vector_store", "graph_store", "fraud_model", "pipeline"):
        assert key in info
    assert info["vector_store"]["clauses"] > 0


def test_claim_listing_and_detail(client):
    claims = client.get("/api/claims").json()
    assert len(claims) == 1
    detail = client.get(f"/api/claims/{claims[0]['id']}").json()
    assert detail["customer_name"] == "Test Claimant"
    assert detail["policy_number"] == "POL-TEST-1"


def test_investigation_produces_a_full_audit_trail(client):
    claim_id = client.get("/api/claims").json()[0]["id"]
    inv = client.post(f"/api/claims/{claim_id}/investigate", json={"force": True}).json()

    assert inv["status"] == "completed"
    assert inv["decision"] in {"approve", "review", "reject"}
    assert 0.0 <= inv["fraud_score"] <= 1.0
    assert inv["rationale"]

    names = [r["agent_name"] for r in inv["agent_runs"]]
    assert "intake" in names and "adjudication" in names
    assert all(r["status"] in {"ok", "skipped"} for r in inv["agent_runs"]), inv["agent_runs"]

    # Coverage must cite clauses, never decide from memory.
    clauses = [e for e in inv["evidence"] if e["type"] == "policy_clause"]
    assert clauses, "coverage decision must be backed by retrieved clauses"


def test_a_clean_well_documented_claim_is_not_rejected(client):
    claim_id = client.get("/api/claims").json()[0]["id"]
    inv = client.post(f"/api/claims/{claim_id}/investigate", json={"force": True}).json()
    assert inv["decision"] != "reject"


def test_graph_and_stats_endpoints(client):
    assert "nodes" in client.get("/api/graph/stats").json()
    stats = client.get("/api/stats").json()
    assert stats["total_claims"] == 1
    assert stats["investigated"] >= 1


def test_unknown_claim_returns_404(client):
    res = client.get("/api/claims/does-not-exist")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"

"""End-to-end tests for the investigation pipeline.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.agents.decision import blend_score  # noqa: E402
from app.agents.pipeline import route_after_anomaly, route_after_intake  # noqa: E402
from app.agents.state import new_state, signal  # noqa: E402
from app.services.documents import extract_fields  # noqa: E402
from app.services.features import FEATURE_NAMES, build_feature_row, to_vector  # noqa: E402
from app.services.graphstore import GraphClaim, InMemoryGraphStore  # noqa: E402
from app.services.ml import get_scorer  # noqa: E402
from app.services.vectorstore import load_policy_chunks  # noqa: E402


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------- features
def test_feature_vector_is_stable_and_complete():
    ctx = {
        "claimed_amount": 5000,
        "estimated_repair_cost": 4000,
        "coverage_limit": 20000,
        "annual_premium": 600,
        "deductible": 500,
        "incident_date": _now() - timedelta(days=10),
        "reported_date": _now(),
        "policy_start_date": _now() - timedelta(days=400),
        "incident_type": "collision",
        "incident_description": "Rear-ended at a junction by a van that failed to stop in time.",
    }
    row = build_feature_row(ctx)
    assert set(FEATURE_NAMES).issubset(row.keys())
    vec = to_vector(row)
    assert len(vec) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vec)
    # reporting delay is 10 days, not negative
    assert row["reporting_delay_days"] == pytest.approx(10, abs=1)


def test_missing_context_does_not_crash_feature_builder():
    row = build_feature_row({})
    assert len(to_vector(row)) == len(FEATURE_NAMES)


# --------------------------------------------------------------------- ml
def test_scorer_returns_probability_and_attribution():
    result = get_scorer().score(
        {
            "claimed_amount": 30000,
            "estimated_repair_cost": 9000,
            "coverage_limit": 25000,
            "annual_premium": 500,
            "incident_date": _now() - timedelta(days=2),
            "reported_date": _now(),
            "policy_start_date": _now() - timedelta(days=12),
            "witnesses": 0,
            "police_report": False,
            "incident_description": "Car was damaged.",
        }
    )
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["top_features"], "expected feature attributions"


def test_obvious_fraud_scores_above_obvious_clean():
    scorer = get_scorer()
    base = {
        "coverage_limit": 25000,
        "annual_premium": 700,
        "incident_type": "collision",
        "estimated_repair_cost": 4000,
    }
    clean = scorer.score(
        {
            **base,
            "claimed_amount": 4100,
            "incident_date": _now() - timedelta(days=30),
            "reported_date": _now() - timedelta(days=29),
            "policy_start_date": _now() - timedelta(days=900),
            "witnesses": 2,
            "police_report": True,
            "document_count": 3,
            "has_invoice": True,
            "incident_description": "Rear-ended at a red light; the other driver admitted fault "
            "and we exchanged details at the scene.",
        }
    )["fraud_probability"]
    dodgy = scorer.score(
        {
            **base,
            "claimed_amount": 22000,
            "incident_date": _now() - timedelta(days=40),
            "reported_date": _now(),
            "policy_start_date": _now() - timedelta(days=48),
            "witnesses": 0,
            "police_report": False,
            "document_count": 0,
            "has_invoice": False,
            "vendor_watchlisted": True,
            "shared_account_degree": 3,
            "ring_size": 6,
            "ring_cohesion": 0.8,
            "incident_description": "Car was damaged.",
        }
    )["fraud_probability"]
    assert dodgy > clean


# ------------------------------------------------------------------ graph
def _graph_claim(i: int, **kwargs) -> GraphClaim:
    defaults = dict(
        claim_id=f"c{i}",
        claim_number=f"CLM-{i}",
        customer_id=f"cust{i}",
        customer_name=f"Customer {i}",
        claimed_amount=1000.0 * i,
    )
    defaults.update(kwargs)
    return GraphClaim(**defaults)  # type: ignore[arg-type]


def test_graph_detects_a_ring_sharing_a_bank_account():
    store = InMemoryGraphStore()
    claims = [_graph_claim(i, bank_account_hash="shared") for i in range(1, 5)]
    claims.append(_graph_claim(99))  # unrelated
    store.rebuild(claims)

    ring = store.ring_for_claim("c1")
    assert ring["size"] == 4
    assert "SHARED_BANK_ACCOUNT" in ring["shared_attributes"]
    assert store.ring_for_claim("c99")["size"] in (0, 1)

    rings = store.list_rings(min_size=3)
    assert len(rings) == 1


def test_popular_vendor_does_not_become_a_ring():
    """A garage with 40 customers is busy, not criminal."""
    store = InMemoryGraphStore()
    store.rebuild([_graph_claim(i, vendor_id="busy-garage") for i in range(1, 41)])
    assert store.graph.number_of_edges() == 0


# -------------------------------------------------------------- documents
def test_invoice_field_extraction():
    text = """Dubois Carrosserie
    Invoice No: INV-88213
    Service Date: 12/04/2025
    Registration: KA01AB1234
    Total Amount: 12,450.00
    """
    fields = extract_fields(text, "invoice")
    assert fields["invoice_number"] == "INV-88213"
    assert fields["total_amount"] == 12450.0
    assert fields["service_date"] == "2025-04-12"
    assert fields["registration"] == "KA01AB1234"


# ------------------------------------------------------------------- rag
def test_policy_corpus_parses_into_clauses():
    chunks = load_policy_chunks()
    assert len(chunks) > 20
    ids = {c.clause_id for c in chunks}
    assert "MC-4.4" in ids  # the fraud clause a decline has to cite
    assert all(c.text.strip() for c in chunks)


# -------------------------------------------------------------- decision
def test_blend_is_monotonic_in_both_inputs():
    low, _ = blend_score(0.1, [])
    high, _ = blend_score(0.9, [])
    assert high > low

    with_signals, pressure = blend_score(
        0.1, [signal("test", "X", "d", "high", 0.3), signal("test", "Y", "d", "high", 0.3)]
    )
    assert with_signals > low
    assert 0.0 < pressure < 1.0


def test_blend_stays_in_range_under_many_signals():
    score, _ = blend_score(1.0, [signal("t", "X", "d", "high", 0.5) for _ in range(20)])
    assert 0.0 <= score <= 1.0


# --------------------------------------------------------------- routing
def test_unusable_filing_short_circuits_to_adjudication():
    state = new_state("x")
    state["intake"] = {"completeness_score": 0.1}
    assert route_after_intake(state) == "adjudication"


def test_normal_filing_continues_through_the_graph():
    state = new_state("x")
    state["intake"] = {"completeness_score": 0.9}
    assert route_after_intake(state) == "document_analysis"


def test_isolated_low_risk_claim_skips_the_network_agent():
    state = new_state("x")
    state["anomaly"] = {"fraud_probability": 0.05}
    state["_graph_context"] = {"neighbours": []}  # type: ignore[typeddict-unknown-key]
    assert route_after_anomaly(state) == "adjudication"


def test_connected_claim_always_visits_the_network_agent():
    state = new_state("x")
    state["anomaly"] = {"fraud_probability": 0.05}
    state["_graph_context"] = {"neighbours": [{"claim_id": "y"}]}  # type: ignore[typeddict-unknown-key]
    assert route_after_anomaly(state) == "network"

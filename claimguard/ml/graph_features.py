"""Build the relationship graph from the CSV and derive per-claim graph features.

Shared by training and evaluation. The live API derives the same features from
the same graph store, so a claim scored offline and the same claim scored
through the API get identical graph inputs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator

import _bootstrap  # noqa: F401  (sys.path side-effect)

from app.services.graphstore import GraphClaim, InMemoryGraphStore


def read_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def iter_rows(csv_path: Path) -> Iterator[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def _week(iso: str) -> str | None:
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso)
        y, w, _ = dt.isocalendar()
        return f"{y}-W{w:02d}"
    except Exception:
        return None


def build_graph(rows: list[dict[str, Any]]) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.rebuild(
        [
            GraphClaim(
                claim_id=r["claim_id"],
                claim_number=r["claim_number"],
                customer_id=r["customer_id"],
                customer_name=r["customer_name"],
                phone=r.get("phone") or None,
                bank_account_hash=r.get("bank_account_hash") or None,
                address=r.get("address") or None,
                vehicle_id=r.get("vehicle_id") or None,
                vendor_id=r.get("vendor_id") or None,
                vendor_name=r.get("vendor_name") or None,
                postal_code=r.get("incident_postal_code") or None,
                incident_week=_week(r.get("incident_date", "")),
                claimed_amount=float(r.get("claimed_amount") or 0.0),
            )
            for r in rows
        ]
    )
    return store


def graph_feature_table(store: InMemoryGraphStore) -> dict[str, dict[str, float]]:
    """Per-claim graph features for every node, computed component-wise.

    Done in one pass over connected components rather than per claim: on a
    150k-claim graph the per-claim version is O(n * component) and takes
    minutes; this takes seconds.
    """
    import networkx as nx

    features: dict[str, dict[str, float]] = {}
    graph = store.graph

    for component in nx.connected_components(graph):
        sub = graph.subgraph(component)
        edges = [(u, v, k) for u, v, d in sub.edges(data=True) for k in d["kinds"]]
        size = len(component)
        max_edges = size * (size - 1) / 2 if size > 1 else 1
        from app.services.graphstore import LINK_WEIGHTS

        weight = sum(LINK_WEIGHTS.get(k, 0.3) for _, _, k in edges)
        cohesion = min(1.0, weight / max_edges) if size > 1 else 0.0

        for node in component:
            phone_deg = account_deg = vendor_deg = 0
            amounts = []
            for nb in sub.neighbors(node):
                kinds = sub[node][nb]["kinds"]
                if "SHARED_PHONE" in kinds:
                    phone_deg += 1
                if "SHARED_BANK_ACCOUNT" in kinds:
                    account_deg += 1
                if "SHARED_VENDOR" in kinds:
                    vendor_deg += 1
                amt = sub.nodes[nb].get("claimed_amount")
                if amt:
                    amounts.append(float(amt))
            features[node] = {
                "ring_size": float(size),
                "ring_cohesion": float(cohesion),
                "shared_phone_degree": float(phone_deg),
                "shared_account_degree": float(account_deg),
                "vendor_degree": float(vendor_deg),
                "neighbour_mean_amount": (sum(amounts) / len(amounts)) if amounts else 0.0,
            }

    # Isolated nodes.
    for node in graph.nodes:
        features.setdefault(
            node,
            {
                "ring_size": 0.0,
                "ring_cohesion": 0.0,
                "shared_phone_degree": 0.0,
                "shared_account_degree": 0.0,
                "vendor_degree": 0.0,
                "neighbour_mean_amount": 0.0,
            },
        )
    return features


def row_to_context(row: dict[str, Any], graph_feats: dict[str, float] | None = None) -> dict:
    """CSV row -> the context dict ``build_feature_row`` expects."""
    ctx: dict[str, Any] = {
        "claimed_amount": float(row.get("claimed_amount") or 0),
        "estimated_repair_cost": float(row.get("estimated_repair_cost") or 0),
        "coverage_limit": float(row.get("coverage_limit") or 25000),
        "annual_premium": float(row.get("annual_premium") or 1),
        "deductible": float(row.get("deductible") or 0),
        "incident_date": row.get("incident_date"),
        "reported_date": row.get("reported_date"),
        "policy_start_date": row.get("policy_start_date"),
        "incident_type": row.get("incident_type"),
        "incident_description": row.get("incident_description"),
        "witnesses": int(float(row.get("witnesses") or 0)),
        "police_report": int(float(row.get("police_report") or 0)),
        "injury_claimed": int(float(row.get("injury_claimed") or 0)),
        "prior_claims_12m": int(float(row.get("prior_claims_12m") or 0)),
        "customer_tenure_months": int(float(row.get("tenure_months") or 0)),
        "vehicle_age": max(0, 2026 - int(float(row.get("year") or 2026))),
        "vehicle_power": int(float(row.get("vehicle_power") or 0)),
        "bonus_malus": int(float(row.get("bonus_malus") or 50)),
        "document_count": int(float(row.get("document_count") or 0)),
        "has_invoice": bool(int(float(row.get("has_invoice") or 0))),
        "vendor_watchlisted": str(row.get("vendor_watchlisted")).lower() in ("true", "1"),
        "vendor_claims_serviced": int(float(row.get("vendor_claims_serviced") or 0)),
        "vendor_avg_invoice": float(row.get("vendor_avg_invoice") or 0),
    }
    if graph_feats:
        ctx.update(graph_feats)
    return ctx


def default_dataset_path() -> Path:
    return Path(_bootstrap.DATA_DIR) / "claims_dataset.csv"

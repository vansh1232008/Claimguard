"""Relationship graph over customers, vehicles, claims and vendors.

Organised fraud is a *structural* signal: individually every claim looks fine,
but the same phone number, bank account or repair garage keeps reappearing.
This module builds that graph and finds the connected components.

Backends: Neo4j when NEO4J_URI is set, otherwise an in-process NetworkX graph
built from the relational store. Same interface either way.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Link types, weighted by how strongly they suggest collusion.
LINK_WEIGHTS: dict[str, float] = {
    "SHARED_PHONE": 1.0,
    "SHARED_BANK_ACCOUNT": 1.0,
    "SHARED_VENDOR": 0.6,
    "SHARED_ADDRESS": 0.8,
    "SHARED_VEHICLE": 0.9,
    "SAME_AREA_SAME_WEEK": 0.35,
}


@dataclass
class GraphClaim:
    """Flat projection of a claim, everything the graph needs in one object."""

    claim_id: str
    claim_number: str
    customer_id: str
    customer_name: str
    phone: str | None = None
    bank_account_hash: str | None = None
    address: str | None = None
    vehicle_id: str | None = None
    vendor_id: str | None = None
    vendor_name: str | None = None
    postal_code: str | None = None
    incident_week: str | None = None
    claimed_amount: float = 0.0
    fraud_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _component_cohesion(edges: list[tuple[str, str, str]], n_nodes: int) -> float:
    """Weighted density of the component: 1.0 means every pair is strongly linked."""
    if n_nodes < 2:
        return 0.0
    max_edges = n_nodes * (n_nodes - 1) / 2
    weight = sum(LINK_WEIGHTS.get(kind, 0.3) for _, _, kind in edges)
    return min(1.0, weight / max_edges)


class InMemoryGraphStore:
    """NetworkX-backed graph, rebuilt from the relational store on demand."""

    name = "in-memory"

    def __init__(self) -> None:
        import networkx as nx

        self._nx = nx
        self.graph = nx.Graph()
        self._claims: dict[str, GraphClaim] = {}

    # -- build ------------------------------------------------------------
    def rebuild(self, claims: list[GraphClaim]) -> dict[str, Any]:
        self.graph = self._nx.Graph()
        self._claims = {c.claim_id: c for c in claims}

        buckets: dict[str, dict[str, list[str]]] = {
            "SHARED_PHONE": defaultdict(list),
            "SHARED_BANK_ACCOUNT": defaultdict(list),
            "SHARED_ADDRESS": defaultdict(list),
            "SHARED_VENDOR": defaultdict(list),
            "SHARED_VEHICLE": defaultdict(list),
            "SAME_AREA_SAME_WEEK": defaultdict(list),
        }

        for c in claims:
            self.graph.add_node(
                c.claim_id,
                claim_number=c.claim_number,
                customer_name=c.customer_name,
                claimed_amount=c.claimed_amount,
                fraud_score=c.fraud_score,
            )
            if c.phone:
                buckets["SHARED_PHONE"][c.phone].append(c.claim_id)
            if c.bank_account_hash:
                buckets["SHARED_BANK_ACCOUNT"][c.bank_account_hash].append(c.claim_id)
            if c.address:
                buckets["SHARED_ADDRESS"][c.address].append(c.claim_id)
            if c.vendor_id:
                buckets["SHARED_VENDOR"][c.vendor_id].append(c.claim_id)
            if c.vehicle_id:
                buckets["SHARED_VEHICLE"][c.vehicle_id].append(c.claim_id)
            if c.postal_code and c.incident_week:
                buckets["SAME_AREA_SAME_WEEK"][f"{c.postal_code}|{c.incident_week}"].append(
                    c.claim_id
                )

        for kind, groups in buckets.items():
            for value, members in groups.items():
                uniq = sorted(set(members))
                if len(uniq) < 2:
                    continue
                # A very large bucket is a popular garage, not a ring: cap it.
                if kind in ("SHARED_VENDOR", "SAME_AREA_SAME_WEEK") and len(uniq) > 25:
                    continue
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        a, b = uniq[i], uniq[j]
                        if self.graph.has_edge(a, b):
                            self.graph[a][b]["kinds"].add(kind)
                            self.graph[a][b]["weight"] = max(
                                self.graph[a][b]["weight"], LINK_WEIGHTS.get(kind, 0.3)
                            )
                        else:
                            self.graph.add_edge(
                                a,
                                b,
                                kinds={kind},
                                weight=LINK_WEIGHTS.get(kind, 0.3),
                                value=str(value)[:64],
                            )
        return {
            "backend": self.name,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }

    # -- query ------------------------------------------------------------
    def neighbours(self, claim_id: str, depth: int = 1) -> list[dict[str, Any]]:
        if claim_id not in self.graph:
            return []
        out = []
        seen = {claim_id}
        frontier = [claim_id]
        for _ in range(max(1, depth)):
            nxt = []
            for node in frontier:
                for nb in self.graph.neighbors(node):
                    if nb in seen:
                        continue
                    seen.add(nb)
                    nxt.append(nb)
                    edge = self.graph[node][nb]
                    claim = self._claims.get(nb)
                    out.append(
                        {
                            "claim_id": nb,
                            "claim_number": self.graph.nodes[nb].get("claim_number"),
                            "customer_name": self.graph.nodes[nb].get("customer_name"),
                            "claimed_amount": self.graph.nodes[nb].get("claimed_amount"),
                            "fraud_score": self.graph.nodes[nb].get("fraud_score"),
                            "relationship": "+".join(sorted(edge["kinds"])),
                            "shared_value": edge.get("value"),
                            "weight": edge.get("weight"),
                            "vendor_name": claim.vendor_name if claim else None,
                        }
                    )
            frontier = nxt
        out.sort(key=lambda d: -(d.get("weight") or 0))
        return out

    def ring_for_claim(self, claim_id: str) -> dict[str, Any]:
        if claim_id not in self.graph:
            return {"size": 0, "shared_attributes": [], "cohesion": 0.0, "members": []}
        component = self._nx.node_connected_component(self.graph, claim_id)
        sub = self.graph.subgraph(component)
        edges = [(u, v, k) for u, v, d in sub.edges(data=True) for k in d["kinds"]]
        kinds = sorted({k for _, _, k in edges})
        members = [
            {
                "claim_id": n,
                "claim_number": sub.nodes[n].get("claim_number"),
                "customer_name": sub.nodes[n].get("customer_name"),
                "claimed_amount": sub.nodes[n].get("claimed_amount", 0.0),
                "fraud_score": sub.nodes[n].get("fraud_score"),
            }
            for n in component
        ]
        return {
            "ring_id": f"comp-{min(component)[:8]}",
            "size": len(component),
            "shared_attributes": kinds,
            "cohesion": round(_component_cohesion(edges, len(component)), 4),
            "total_exposure": round(sum(m["claimed_amount"] or 0 for m in members), 2),
            "members": members,
        }

    def list_rings(self, min_size: int = 3) -> list[dict[str, Any]]:
        rings = []
        for component in self._nx.connected_components(self.graph):
            if len(component) < min_size:
                continue
            rings.append(self.ring_for_claim(next(iter(component))))
        rings.sort(key=lambda r: (-r["size"], -r["total_exposure"]))
        return rings

    def subgraph_payload(self, claim_id: str, depth: int = 2) -> dict[str, Any]:
        """Nodes + links shaped for the frontend force-directed view."""
        if claim_id not in self.graph:
            return {"nodes": [], "links": []}
        seen = {claim_id}
        frontier = [claim_id]
        for _ in range(depth):
            nxt = []
            for node in frontier:
                for nb in self.graph.neighbors(node):
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
        sub = self.graph.subgraph(seen)
        return {
            "nodes": [
                {
                    "id": n,
                    "label": sub.nodes[n].get("claim_number"),
                    "customer": sub.nodes[n].get("customer_name"),
                    "amount": sub.nodes[n].get("claimed_amount"),
                    "fraud_score": sub.nodes[n].get("fraud_score"),
                    "focus": n == claim_id,
                }
                for n in sub.nodes
            ],
            "links": [
                {
                    "source": u,
                    "target": v,
                    "kinds": sorted(d["kinds"]),
                    "weight": d["weight"],
                    "value": d.get("value"),
                }
                for u, v, d in sub.edges(data=True)
            ],
        }

    def stats(self) -> dict[str, Any]:
        rings = self.list_rings()
        return {
            "backend": self.name,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "rings": len(rings),
            "largest_ring": rings[0]["size"] if rings else 0,
        }


class Neo4jGraphStore(InMemoryGraphStore):
    """Writes the same graph into Neo4j and reads rings back with Cypher.

    Subclasses the in-memory store so analytics keep working even while the
    Neo4j write path is the source of truth for the UI's graph explorer.
    """

    name = "neo4j"

    def __init__(self) -> None:
        super().__init__()
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self._db = settings.neo4j_database

    def rebuild(self, claims: list[GraphClaim]) -> dict[str, Any]:
        result = super().rebuild(claims)
        with self._driver.session(database=self._db) as session:
            session.run("MATCH (n) DETACH DELETE n")
            session.run(
                """
                UNWIND $rows AS row
                MERGE (cu:Customer {id: row.customer_id})
                  SET cu.name = row.customer_name, cu.phone = row.phone,
                      cu.bank_account = row.bank_account_hash, cu.address = row.address
                MERGE (cl:Claim {id: row.claim_id})
                  SET cl.number = row.claim_number, cl.amount = row.claimed_amount,
                      cl.fraud_score = row.fraud_score, cl.postal_code = row.postal_code
                MERGE (cu)-[:FILED]->(cl)
                FOREACH (_ IN CASE WHEN row.vehicle_id IS NULL THEN [] ELSE [1] END |
                  MERGE (v:Vehicle {id: row.vehicle_id})
                  MERGE (cl)-[:INVOLVES]->(v)
                  MERGE (cu)-[:OWNS]->(v))
                FOREACH (_ IN CASE WHEN row.vendor_id IS NULL THEN [] ELSE [1] END |
                  MERGE (ve:Vendor {id: row.vendor_id})
                    SET ve.name = row.vendor_name
                  MERGE (cl)-[:SERVICED_BY]->(ve))
                """,
                rows=[c.__dict__ for c in claims],
            )
            # Materialise the collusion edges Cypher-side too.
            session.run(
                """
                MATCH (a:Customer), (b:Customer)
                WHERE a.id < b.id AND a.phone IS NOT NULL AND a.phone = b.phone
                MERGE (a)-[:SHARES_PHONE]->(b)
                """
            )
            session.run(
                """
                MATCH (a:Customer), (b:Customer)
                WHERE a.id < b.id AND a.bank_account IS NOT NULL
                  AND a.bank_account = b.bank_account
                MERGE (a)-[:SHARES_BANK_ACCOUNT]->(b)
                """
            )
        result["backend"] = self.name
        return result

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass


_graph_store: InMemoryGraphStore | None = None


def get_graph_store() -> InMemoryGraphStore:
    global _graph_store
    if _graph_store is None:
        if settings.neo4j_uri and settings.neo4j_password:
            try:
                _graph_store = Neo4jGraphStore()
                logger.info("Graph store: Neo4j at %s", settings.neo4j_uri)
            except Exception as exc:
                logger.warning("Neo4j unavailable (%s); using in-memory graph", exc)
                _graph_store = InMemoryGraphStore()
        else:
            _graph_store = InMemoryGraphStore()
            logger.info("Graph store: in-memory (NetworkX)")
    return _graph_store

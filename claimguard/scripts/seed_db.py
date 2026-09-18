"""Seed the application database.

    python scripts/seed_db.py --claims 400

Takes a slice of the generated dataset (or generates one on the fly if the CSV
is missing) and loads customers, vehicles, vendors, policies and claims into
the relational store, then builds the relationship graph. Ring members are kept
together so the graph view has something real to show.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "ml"))

from app.database import init_db, session_scope  # noqa: E402
from app.models import (  # noqa: E402
    AgentRun,
    Claim,
    ClaimDocument,
    Customer,
    Investigation,
    Policy,
    Vehicle,
    Vendor,
)
from app.services.graph_sync import refresh_graph  # noqa: E402

INVOICE_TEMPLATE = """{vendor_name}
GSTIN: 29ABCDE{n:04d}F1Z5
Invoice No: INV-{invoice_no}
Service Date: {service_date}
Registration: {registration}

Description                     Amount
Parts and panels                {parts:.2f}
Paint and materials             {paint:.2f}
Labour                          {labour:.2f}
--------------------------------------
Total Amount: {total:.2f}
"""


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def ensure_dataset(path: Path, n_claims: int) -> Path:
    if path.exists():
        return path
    print(f"{path} not found - generating a small dataset ({max(n_claims * 4, 4000)} claims) ...")
    from generate_dataset import Generator, write_csv

    gen = Generator(n_claims=max(n_claims * 4, 4000), n_rings=max(12, n_claims // 25))
    write_csv(gen.generate(), path)
    return path


def pick_rows(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Keep whole rings together, then fill up with ordinary claims."""
    ringed = [r for r in rows if r.get("fraud_ring_id")]
    plain = [r for r in rows if not r.get("fraud_ring_id")]

    by_ring: dict[str, list[dict]] = {}
    for r in ringed:
        by_ring.setdefault(r["fraud_ring_id"], []).append(r)

    chosen: list[dict] = []
    for ring_rows in sorted(by_ring.values(), key=len, reverse=True):
        if len(chosen) >= n * 0.35:
            break
        chosen.extend(ring_rows[:8])

    rng.shuffle(plain)
    chosen.extend(plain[: max(0, n - len(chosen))])
    rng.shuffle(chosen)
    return chosen[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=int, default=400)
    parser.add_argument("--data", default="data/claims_dataset.csv")
    parser.add_argument("--documents", type=float, default=0.6,
                        help="fraction of claims that get a generated invoice")
    parser.add_argument("--reset", action="store_true", help="wipe existing rows first")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from graph_features import read_rows

    rng = random.Random(args.seed)
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    ensure_dataset(data_path, args.claims)

    rows = pick_rows(read_rows(data_path), args.claims, rng)
    print(f"Seeding {len(rows)} claims ...")

    init_db()
    doc_dir = REPO_ROOT / "data" / "uploads"
    doc_dir.mkdir(parents=True, exist_ok=True)

    with session_scope() as db:
        if args.reset:
            # Bulk deletes bypass the ORM's cascade rules, so child tables are
            # listed explicitly and deleted first. Leaving them out orphans the
            # old investigations, which then keep showing up in /api/stats.
            for model in (
                AgentRun,
                Investigation,
                ClaimDocument,
                Claim,
                Policy,
                Vehicle,
                Customer,
                Vendor,
            ):
                db.query(model).delete()
            db.flush()

        customers: dict[str, Customer] = {}
        vendors: dict[str, Vendor] = {}
        vehicles: dict[str, Vehicle] = {}
        policies: dict[str, Policy] = {}

        for row in rows:
            cust = customers.get(row["customer_id"])
            if cust is None:
                cust = Customer(
                    external_id=row["customer_id"],
                    full_name=row["customer_name"],
                    email=f"{row['customer_name'].split()[0].lower()}@example.test",
                    phone=row["phone"],
                    address=row["address"],
                    postal_code=row["postal_code"],
                    region=row["region"],
                    bank_account_hash=row["bank_account_hash"],
                    tenure_months=int(float(row["tenure_months"])),
                )
                db.add(cust)
                db.flush()
                customers[row["customer_id"]] = cust

            vendor = vendors.get(row["vendor_id"])
            if vendor is None:
                vendor = Vendor(
                    external_id=row["vendor_id"],
                    name=row["vendor_name"],
                    vendor_type=row.get("vendor_type", "garage"),
                    postal_code=row["postal_code"],
                    watchlisted=str(row["vendor_watchlisted"]).lower() in ("true", "1"),
                    avg_invoice_amount=float(row["vendor_avg_invoice"]),
                    claims_serviced=int(float(row["vendor_claims_serviced"])),
                )
                db.add(vendor)
                db.flush()
                vendors[row["vendor_id"]] = vendor

            vehicle = vehicles.get(row["vehicle_id"])
            if vehicle is None:
                vehicle = Vehicle(
                    vin=row["vin"],
                    make=row["make"],
                    model=row["model"],
                    year=int(float(row["year"])),
                    vehicle_power=int(float(row["vehicle_power"])),
                    vehicle_gas=row["vehicle_gas"],
                    declared_value=float(row["declared_value"]),
                )
                db.add(vehicle)
                db.flush()
                vehicles[row["vehicle_id"]] = vehicle

            policy = policies.get(row["policy_number"])
            if policy is None:
                policy = Policy(
                    policy_number=row["policy_number"],
                    customer_id=cust.id,
                    vehicle_id=vehicle.id,
                    status="active",
                    start_date=_dt(row["policy_start_date"]),
                    end_date=_dt(row["policy_end_date"]),
                    annual_premium=float(row["annual_premium"]),
                    coverage_limit=float(row["coverage_limit"]),
                    deductible=float(row["deductible"]),
                    bonus_malus=int(float(row["bonus_malus"])),
                    exclusions=[
                        "driving_under_influence",
                        "unauthorised_modification",
                        "racing",
                        "water_ingress",
                    ],
                )
                db.add(policy)
                db.flush()
                policies[row["policy_number"]] = policy

            claim = Claim(
                claim_number=row["claim_number"],
                customer_id=cust.id,
                policy_id=policy.id,
                vehicle_id=vehicle.id,
                vendor_id=vendor.id,
                incident_type=row["incident_type"],
                incident_date=_dt(row["incident_date"]),
                reported_date=_dt(row["reported_date"]),
                incident_description=row["incident_description"],
                incident_postal_code=row["incident_postal_code"],
                claimed_amount=float(row["claimed_amount"]),
                estimated_repair_cost=float(row["estimated_repair_cost"]),
                witnesses=int(float(row["witnesses"])),
                police_report=bool(int(float(row["police_report"]))),
                injury_claimed=bool(int(float(row["injury_claimed"]))),
                prior_claims_12m=int(float(row["prior_claims_12m"])),
                status="submitted",
                is_fraud_label=bool(int(float(row["is_fraud"]))),
                fraud_ring_id=row.get("fraud_ring_id") or None,
            )
            db.add(claim)
            db.flush()

            if rng.random() < args.documents:
                total = float(row["claimed_amount"])
                # Padded claims get an invoice that disagrees with the claim form,
                # which is exactly what the document agent is built to catch.
                if bool(int(float(row["is_fraud"]))) and rng.random() < 0.55:
                    total = float(row["estimated_repair_cost"])
                parts = round(total * 0.52, 2)
                paint = round(total * 0.18, 2)
                labour = round(total - parts - paint, 2)
                text = INVOICE_TEMPLATE.format(
                    vendor_name=vendor.name,
                    n=rng.randint(0, 9999),
                    invoice_no=f"{rng.randint(10000, 99999)}",
                    service_date=str(row["incident_date"])[:10],
                    registration=row["vin"][:10].upper(),
                    parts=parts,
                    paint=paint,
                    labour=labour,
                    total=total,
                )
                path = doc_dir / f"{claim.claim_number}_invoice.txt"
                path.write_text(text, encoding="utf-8")

                from app.services.documents import process_document

                processed = process_document(path, "invoice")
                db.add(
                    ClaimDocument(
                        claim_id=claim.id,
                        doc_type="invoice",
                        filename=path.name,
                        storage_path=str(path),
                        mime_type="text/plain",
                        raw_text=processed["raw_text"],
                        extracted_fields=processed["extracted_fields"],
                    )
                )

        db.flush()
        stats = refresh_graph(db)

    print(
        f"Seeded {len(rows)} claims, {len(customers)} customers, {len(vendors)} vendors.\n"
        f"Graph: {stats['nodes']} nodes / {stats['edges']} edges ({stats['backend']})"
    )
    print("Next: uvicorn app.main:app --reload --app-dir backend")


if __name__ == "__main__":
    main()

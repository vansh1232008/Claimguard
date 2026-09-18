"""Synthetic claims dataset generator.

Why synthetic: no public motor-insurance dataset ships with labelled organised
fraud. The portfolio here is the *detection system*, so the data is generated to
have the structure the system is built to find:

* a freMTPL2-shaped population of French motor policies (vehicle power, vehicle
  age, driver age, bonus-malus, region, density) so the marginal distributions
  are realistic rather than uniform noise,
* an individual fraud layer (~8% of claims) whose signal is *behavioural*, not a
  giveaway column: shorter time since inception, later reporting, thinner
  narratives, missing police reports, invoices above estimate,
* an organised layer: N rings whose members deliberately share identity
  attributes (phone, bank account, address) and repair vendors, which is the
  structure the graph agent exists to detect.

Nothing here is drawn from a real claimant. Run:

    python ml/generate_dataset.py --claims 150000 --out data/claims_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REGIONS = [
    "Ile-de-France", "Rhone-Alpes", "Provence-Alpes-Cote-d-Azur", "Nord-Pas-de-Calais",
    "Aquitaine", "Bretagne", "Centre", "Pays-de-la-Loire", "Occitanie", "Normandie",
]
BRANDS = ["Renault", "Peugeot", "Citroen", "Volkswagen", "Ford", "Toyota", "Fiat", "Opel"]
MODELS = {
    "Renault": ["Clio", "Megane", "Captur"],
    "Peugeot": ["208", "308", "3008"],
    "Citroen": ["C3", "C4", "Berlingo"],
    "Volkswagen": ["Polo", "Golf", "Tiguan"],
    "Ford": ["Fiesta", "Focus", "Kuga"],
    "Toyota": ["Yaris", "Corolla", "RAV4"],
    "Fiat": ["Panda", "500", "Tipo"],
    "Opel": ["Corsa", "Astra", "Mokka"],
}
INCIDENT_TYPES = ["collision", "theft", "fire", "vandalism", "glass", "weather", "third_party_injury"]
INCIDENT_WEIGHTS = [0.52, 0.09, 0.03, 0.10, 0.14, 0.09, 0.03]

FIRST_NAMES = [
    "Camille", "Lucas", "Manon", "Hugo", "Emma", "Nathan", "Chloe", "Louis", "Sarah", "Theo",
    "Ines", "Gabriel", "Julie", "Enzo", "Lea", "Raphael", "Elise", "Mathis", "Clara", "Adam",
]
LAST_NAMES = [
    "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand", "Leroy",
    "Moreau", "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David", "Bertrand", "Roux",
]
STREETS = ["Rue de la Paix", "Avenue Victor Hugo", "Boulevard Saint-Michel", "Rue des Lilas",
           "Allee des Chenes", "Rue Gambetta", "Avenue Jean Jaures", "Place du Marche"]

DETAILED_DESCRIPTIONS = [
    "Rear-ended at a red light on the {street}; the other driver admitted fault at the scene and we exchanged details.",
    "Lost control on a wet surface near {street} and struck the kerb, damaging the front-left wheel arch and bumper.",
    "Parked vehicle was hit overnight outside {street}; wing mirror snapped off and the driver-side door is creased.",
    "Stone thrown up by a lorry on the motorway cracked the windscreen across the driver's line of sight.",
    "Hailstorm on the evening of the incident left dents across the roof and bonnet; photographs taken next morning.",
    "Reversing out of a parking bay at {street} when another car crossed behind; contact to the rear bumper and tailgate.",
    "Water entered the cabin during flooding on {street}; carpets and door cards are soaked and the electrics are faulty.",
    "Vehicle was broken into outside {street}; the quarter-light glass was smashed and the dashboard unit removed.",
]
TERSE_DESCRIPTIONS = [
    "Someone hit my car and drove off.",
    "The accident happened and there was damage.",
    "Car was damaged badly, need full repair.",
    "Major damage occurred to the vehicle.",
    "Hit my car at night, nobody was around.",
    "Vehicle got damaged, please settle full amount.",
    "Collision at the junction, other party at fault.",
    "Damage to the front of the car after an accident.",
]
# Both pools are used by both classes. Fraudulent claims lean terse and genuine
# ones lean detailed, but the overlap is large on purpose: if the text pool gave
# the label away, the model would learn the generator instead of the problem.
P_TERSE_FRAUD = 0.55
P_TERSE_GENUINE = 0.30


def _hash(value: str) -> str:
    return hashlib.blake2b(value.encode(), digest_size=8).hexdigest()


@dataclass
class Generator:
    n_claims: int = 150_000
    fraud_rate: float = 0.08
    n_rings: int = 120
    ring_share_of_fraud: float = 0.35
    # Share of fraudulent claims that present as completely ordinary, and share
    # of genuine claims that present as suspicious. These two numbers set the
    # ceiling on achievable recall and precision.
    clean_looking_fraud: float = 0.22
    suspicious_looking_genuine: float = 0.14
    seed: int = 42
    start: datetime = field(
        default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)
    )
    horizon_days: int = 730

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.n_customers = max(1000, int(self.n_claims * 0.55))
        self.n_vendors = max(40, int(self.n_claims * 0.004))

    # ------------------------------------------------------------ entities
    def make_vendors(self) -> list[dict]:
        vendors = []
        for i in range(self.n_vendors):
            watch = self.rng.random() < 0.06
            vendors.append(
                {
                    "vendor_id": f"VND{i:05d}",
                    "vendor_name": f"{self.rng.choice(LAST_NAMES)} "
                    f"{self.rng.choice(['Auto', 'Motors', 'Carrosserie', 'Garage'])}",
                    "vendor_type": self.rng.choice(["garage", "garage", "garage", "clinic"]),
                    "vendor_watchlisted": watch,
                    "vendor_claims_serviced": self.rng.randint(5, 400),
                    "vendor_avg_invoice": round(self.rng.uniform(900, 4200), 2),
                }
            )
        return vendors

    def make_customer(self, idx: int) -> dict:
        name = f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"
        region = self.rng.choice(REGIONS)
        postal = f"{self.rng.randint(1, 95):02d}{self.rng.randint(100, 999)}"
        return {
            "customer_id": f"CUS{idx:06d}",
            "customer_name": name,
            "phone": f"+336{self.rng.randint(10_000_000, 99_999_999)}",
            "bank_account_hash": _hash(f"acct-{idx}-{self.rng.random()}"),
            # The address string has to be near-unique. With a small address
            # space, thousands of unrelated customers collide and the graph
            # fills with meaningless SHARED_ADDRESS edges that drown the real
            # ring structure.
            "address": (
                f"{self.rng.randint(1, 180)}{self.rng.choice(['', ' bis', ' ter', 'A', 'B'])} "
                f"{self.rng.choice(STREETS)}, {postal} {region}"
            ),
            "postal_code": postal,
            "region": region,
            "tenure_months": self.rng.randint(1, 160),
            "driver_age": self.rng.randint(19, 78),
        }

    def make_policy(self, customer: dict, idx: int) -> dict:
        """Policy and vehicle attributes for a customer.

        The date fields here are placeholders: ``make_claim`` derives the real
        inception date backwards from the incident date (see the note there).
        """
        start = self.start - timedelta(days=self.rng.randint(0, 900))
        power = self.rng.randint(4, 12)
        declared = round(self.rng.uniform(4500, 38000), 2)
        limit = round(declared * self.rng.uniform(0.9, 1.1), 2)
        brand = self.rng.choice(BRANDS)
        return {
            "policy_number": f"POL-{idx:07d}",
            "policy_start_date": start.isoformat(),
            "policy_end_date": (start + timedelta(days=365 * 3)).isoformat(),
            "coverage_limit": limit,
            "deductible": float(self.rng.choice([250, 300, 500, 750, 1000])),
            "annual_premium": round(self.rng.uniform(280, 1450), 2),
            "bonus_malus": int(max(50, min(180, self.rng.gauss(75, 22)))),
            "vehicle_id": f"VEH{idx:06d}",
            "vin": f"VF{self.rng.randint(10**12, 10**13 - 1)}",
            "make": brand,
            "model": self.rng.choice(MODELS[brand]),
            "year": self.rng.randint(2006, 2024),
            "vehicle_power": power,
            "vehicle_gas": self.rng.choice(["Regular", "Diesel"]),
            "declared_value": declared,
        }

    # -------------------------------------------------------------- claims
    def _amount(self, incident_type: str, declared: float, suspicious: bool) -> tuple[float, float]:
        base = {
            "collision": self.rng.uniform(0.04, 0.35),
            "theft": self.rng.uniform(0.55, 1.0),
            "fire": self.rng.uniform(0.4, 1.0),
            "vandalism": self.rng.uniform(0.02, 0.15),
            "glass": self.rng.uniform(0.01, 0.05),
            "weather": self.rng.uniform(0.03, 0.25),
            "third_party_injury": self.rng.uniform(0.2, 0.8),
        }[incident_type]
        estimate = round(max(180.0, declared * base), 2)
        # Genuine claims also exceed the estimate sometimes (supplementary work
        # approved mid-repair), and padded claims are often only modestly padded.
        if suspicious:
            ratio = self.rng.gauss(1.22, 0.26)
        else:
            ratio = self.rng.gauss(1.02, 0.13)
        ratio = max(0.75, min(2.4, ratio))
        claimed = round(estimate * ratio, 2)
        return claimed, estimate

    def make_claim(
        self,
        idx: int,
        customer: dict,
        policy: dict,
        vendor: dict,
        fraudulent: bool,
        ring_id: str | None,
    ) -> dict:
        incident_type = self.rng.choices(INCIDENT_TYPES, INCIDENT_WEIGHTS)[0]
        policy_start = datetime.fromisoformat(policy["policy_start_date"])

        # The behavioural profile is drawn from the label only *most* of the
        # time. A share of fraudulent claims behave impeccably (the ones every
        # real model misses) and a share of genuine claims tick several
        # suspicious boxes by bad luck (the false positives that make precision
        # hard). Without this overlap the dataset is separable and any metric
        # measured on it is meaningless.
        if fraudulent:
            suspicious = self.rng.random() > self.clean_looking_fraud
        else:
            suspicious = self.rng.random() < self.suspicious_looking_genuine

        if suspicious:
            days_in = self.rng.choice(
                [self.rng.randint(3, 60), self.rng.randint(20, 300), self.rng.randint(30, 700)]
            )
            delay = max(0, int(self.rng.gauss(6.0, 8.0)))
            witnesses = self.rng.choices([0, 1, 2, 3], [0.50, 0.30, 0.14, 0.06])[0]
            police = self.rng.random() < 0.40
            docs = self.rng.choices([0, 1, 2, 3], [0.18, 0.37, 0.30, 0.15])[0]
            terse = self.rng.random() < P_TERSE_FRAUD
            injury = self.rng.random() < 0.18
        else:
            days_in = self.rng.randint(5, min(self.horizon_days, 900))
            delay = max(0, int(self.rng.gauss(3.0, 4.0)))
            witnesses = self.rng.choices([0, 1, 2, 3], [0.32, 0.36, 0.22, 0.10])[0]
            police = self.rng.random() < 0.58
            docs = self.rng.choices([0, 1, 2, 3], [0.08, 0.30, 0.38, 0.24])[0]
            terse = self.rng.random() < P_TERSE_GENUINE
            injury = self.rng.random() < 0.09

        description = (
            self.rng.choice(TERSE_DESCRIPTIONS)
            if terse
            else self.rng.choice(DETAILED_DESCRIPTIONS).format(street=self.rng.choice(STREETS))
        )

        # Incident dates are drawn uniformly across the observation window and
        # the policy inception is derived backwards from them. Doing it the
        # other way round (inception first, incident = inception + days_in)
        # pushes fraudulent claims towards the start of the calendar, because
        # they sit closer to inception - and a time-based split then puts almost
        # no fraud in the test fold. The signal must live in
        # days_since_policy_start, not in the calendar date.
        incident_date = self.start + timedelta(
            days=self.rng.randint(0, self.horizon_days),
            hours=self.rng.randint(0, 23),
        )
        policy_start = incident_date - timedelta(days=days_in)
        reported_date = incident_date + timedelta(days=delay)
        claimed, estimate = self._amount(incident_type, policy["declared_value"], suspicious)

        policy = {
            **policy,
            "policy_number": f"POL-{idx:07d}",
            "policy_start_date": policy_start.isoformat(),
            "policy_end_date": (policy_start + timedelta(days=1095)).isoformat(),
        }

        return {
            "claim_id": f"CLM{idx:07d}",
            "claim_number": f"CLM-2026-{idx:07d}",
            **customer,
            **policy,
            **vendor,
            "incident_type": incident_type,
            "incident_date": incident_date.isoformat(),
            "reported_date": reported_date.isoformat(),
            "incident_description": description,
            "incident_postal_code": customer["postal_code"],
            "claimed_amount": claimed,
            "estimated_repair_cost": estimate,
            "witnesses": witnesses,
            "police_report": int(police),
            "injury_claimed": int(injury),
            "prior_claims_12m": self.rng.choices([0, 1, 2, 3], [0.68, 0.2, 0.08, 0.04])[0]
            + (1 if suspicious and self.rng.random() < 0.35 else 0),
            "document_count": docs,
            "has_invoice": int(docs > 0),
            "is_fraud": int(fraudulent),
            "fraud_ring_id": ring_id or "",
        }

    # ---------------------------------------------------------------- main
    def generate(self) -> list[dict]:
        vendors = self.make_vendors()
        customers = [self.make_customer(i) for i in range(self.n_customers)]
        policies = [self.make_policy(customers[i], i) for i in range(self.n_customers)]

        n_fraud = int(self.n_claims * self.fraud_rate)
        n_ring_fraud = int(n_fraud * self.ring_share_of_fraud)
        fraud_flags = [True] * n_fraud + [False] * (self.n_claims - n_fraud)
        self.rng.shuffle(fraud_flags)

        # Pre-assign ring membership to the first n_ring_fraud fraudulent claims.
        fraud_positions = [i for i, f in enumerate(fraud_flags) if f]
        ring_positions = set(fraud_positions[:n_ring_fraud])
        ring_assignment: dict[int, str] = {}
        ring_vendor: dict[str, dict] = {}
        ring_identity: dict[str, dict] = {}
        ring_ids = [f"RING-{i:04d}" for i in range(self.n_rings)]
        for i, pos in enumerate(sorted(ring_positions)):
            ring = ring_ids[i % self.n_rings]
            ring_assignment[pos] = ring
            if ring not in ring_vendor:
                ring_vendor[ring] = self.rng.choice(vendors)
                anchor = self.rng.choice(customers)
                ring_identity[ring] = {
                    "phone": anchor["phone"],
                    "bank_account_hash": anchor["bank_account_hash"],
                    "address": anchor["address"],
                    "postal_code": anchor["postal_code"],
                }

        rows: list[dict] = []
        for idx in range(self.n_claims):
            fraudulent = fraud_flags[idx]
            ring = ring_assignment.get(idx)
            ci = self.rng.randrange(self.n_customers)
            customer = dict(customers[ci])
            policy = policies[ci]

            if ring:
                # Ring members deliberately share identity attributes. Which
                # attribute is shared varies, so the graph has mixed edge types.
                shared = ring_identity[ring]
                mode = self.rng.random()
                if mode < 0.45:
                    customer["phone"] = shared["phone"]
                elif mode < 0.75:
                    customer["bank_account_hash"] = shared["bank_account_hash"]
                else:
                    customer["address"] = shared["address"]
                    customer["postal_code"] = shared["postal_code"]
                vendor = ring_vendor[ring]
            else:
                vendor = self.rng.choice(vendors)
                if fraudulent and self.rng.random() < 0.20:
                    vendor = self.rng.choice([v for v in vendors if v["vendor_watchlisted"]] or vendors)

            rows.append(self.make_claim(idx, customer, policy, vendor, fraudulent, ring))
        return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ClaimGuard benchmark dataset")
    parser.add_argument("--claims", type=int, default=150_000)
    parser.add_argument("--fraud-rate", type=float, default=0.08)
    parser.add_argument("--rings", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-looking-fraud", type=float, default=0.22,
                        help="share of fraud that presents as completely ordinary")
    parser.add_argument("--suspicious-looking-genuine", type=float, default=0.14,
                        help="share of genuine claims that tick suspicious boxes")
    parser.add_argument("--out", type=str, default="data/claims_dataset.csv")
    args = parser.parse_args()

    gen = Generator(
        n_claims=args.claims,
        fraud_rate=args.fraud_rate,
        n_rings=args.rings,
        seed=args.seed,
        clean_looking_fraud=args.clean_looking_fraud,
        suspicious_looking_genuine=args.suspicious_looking_genuine,
    )
    rows = gen.generate()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    write_csv(rows, out_path)

    frauds = sum(r["is_fraud"] for r in rows)
    ringed = sum(1 for r in rows if r["fraud_ring_id"])
    print(f"Wrote {len(rows):,} claims to {out_path}")
    print(f"  fraudulent: {frauds:,} ({frauds / len(rows):.1%})")
    print(f"  in rings:   {ringed:,} across {args.rings} rings")
    print(f"  customers:  {gen.n_customers:,}   vendors: {gen.n_vendors:,}")


if __name__ == "__main__":
    main()

"""Run the pipeline over seeded claims and print a readable summary.

    python scripts/run_demo.py --limit 25

Useful as a smoke test (does the whole graph execute?) and as the thing to run
on a screen share when someone asks "show me how it works".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from sqlalchemy import select  # noqa: E402

from app.database import init_db, session_scope  # noqa: E402
from app.models import Claim  # noqa: E402
from app.services.graph_sync import refresh_graph  # noqa: E402
from app.services.investigation import investigate_claim  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--only-fraud", action="store_true",
                        help="only run claims labelled fraudulent in the synthetic data")
    args = parser.parse_args()

    init_db()
    with session_scope() as db:
        refresh_graph(db)
        stmt = select(Claim).where(Claim.status == "submitted")
        if args.only_fraud:
            stmt = stmt.where(Claim.is_fraud_label.is_(True))
        claims = db.execute(stmt.limit(args.limit)).scalars().all()

        if not claims:
            print("No un-investigated claims found. Run: python scripts/seed_db.py")
            return

        print(f"{'claim':<22}{'label':>7}{'score':>8}{'decision':>11}{'payout':>12}{'ms':>7}  agents")
        print("-" * 96)
        correct = 0
        for claim in claims:
            inv = investigate_claim(db, claim.id)
            label = "fraud" if claim.is_fraud_label else "clean"
            flagged = (inv.decision or "review") in ("review", "reject")
            if flagged == bool(claim.is_fraud_label):
                correct += 1
            agents = len(inv.agent_runs)
            print(
                f"{claim.claim_number:<22}{label:>7}{inv.fraud_score or 0:>8.2f}"
                f"{inv.decision or '-':>11}{inv.recommended_payout or 0:>12,.0f}"
                f"{inv.duration_ms:>7}  {agents}"
            )
        print("-" * 96)
        print(
            f"{len(claims)} claims processed. "
            f"Flag agreement with the synthetic labels: {correct}/{len(claims)} "
            f"({correct / len(claims):.0%})."
        )
        print(
            "\nThis is a smoke test, not an evaluation - run ml/evaluate.py for real metrics."
        )


if __name__ == "__main__":
    main()

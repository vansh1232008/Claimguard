"""Feature engineering.

One module, used by BOTH the training scripts and the live pipeline, so the
features a model is trained on are byte-for-byte the features it scores on.
Training/serving skew is the classic way a fraud model quietly stops working.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

BASE_FEATURES: list[str] = [
    "claimed_amount_log",
    "amount_to_limit_ratio",
    "amount_to_premium_ratio",
    "amount_vs_estimate_ratio",
    "deductible_ratio",
    "reporting_delay_days",
    "days_since_policy_start",
    "policy_age_days",
    "incident_dayofweek",
    "incident_is_weekend",
    "incident_month",
    "witnesses",
    "police_report",
    "injury_claimed",
    "prior_claims_12m",
    "customer_tenure_months",
    "vehicle_age",
    "vehicle_power",
    "bonus_malus",
    "description_word_count",
    "description_is_generic",
    "document_count",
    "missing_invoice",
    "vendor_watchlisted",
    "vendor_claims_serviced",
    "vendor_invoice_ratio",
    "incident_type_code",
]

GRAPH_FEATURES: list[str] = [
    "ring_size",
    "ring_cohesion",
    "shared_phone_degree",
    "shared_account_degree",
    "vendor_degree",
    "neighbour_mean_amount_ratio",
]

FEATURE_NAMES: list[str] = BASE_FEATURES + GRAPH_FEATURES

INCIDENT_TYPES = [
    "collision",
    "theft",
    "fire",
    "vandalism",
    "glass",
    "weather",
    "third_party_injury",
]

_GENERIC_PHRASES = (
    "hit my car",
    "accident happened",
    "damage occurred",
    "car was damaged",
    "someone hit",
)


def _as_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _days_between(a: Any, b: Any) -> float:
    da, db = _as_dt(a), _as_dt(b)
    if not da or not db:
        return 0.0
    return (da - db).total_seconds() / 86400.0


def _safe_ratio(num: float, den: float, cap: float = 50.0) -> float:
    if not den:
        return 0.0
    return float(min(cap, num / den))


def build_feature_row(ctx: dict[str, Any]) -> dict[str, float]:
    """Turn a flat claim context into the model's feature vector.

    ``ctx`` keys mirror the columns of the generated dataset and the fields the
    API assembles for a live claim, so one function serves both.
    """
    claimed = float(ctx.get("claimed_amount") or 0.0)
    estimate = float(ctx.get("estimated_repair_cost") or 0.0)
    limit = float(ctx.get("coverage_limit") or 25000.0)
    premium = float(ctx.get("annual_premium") or 1.0)
    deductible = float(ctx.get("deductible") or 0.0)
    incident_dt = _as_dt(ctx.get("incident_date"))
    description = str(ctx.get("incident_description") or "")
    words = description.split()

    row: dict[str, float] = {
        "claimed_amount_log": math.log1p(max(claimed, 0.0)),
        "amount_to_limit_ratio": _safe_ratio(claimed, limit, cap=5.0),
        "amount_to_premium_ratio": _safe_ratio(claimed, premium, cap=200.0),
        "amount_vs_estimate_ratio": _safe_ratio(claimed, estimate or claimed or 1.0, cap=10.0),
        "deductible_ratio": _safe_ratio(deductible, claimed or 1.0, cap=5.0),
        "reporting_delay_days": max(
            0.0, _days_between(ctx.get("reported_date"), ctx.get("incident_date"))
        ),
        "days_since_policy_start": max(
            0.0, _days_between(ctx.get("incident_date"), ctx.get("policy_start_date"))
        ),
        "policy_age_days": max(
            0.0, _days_between(ctx.get("reported_date"), ctx.get("policy_start_date"))
        ),
        "incident_dayofweek": float(incident_dt.weekday()) if incident_dt else 0.0,
        "incident_is_weekend": float(incident_dt.weekday() >= 5) if incident_dt else 0.0,
        "incident_month": float(incident_dt.month) if incident_dt else 0.0,
        "witnesses": float(ctx.get("witnesses") or 0),
        "police_report": float(bool(ctx.get("police_report"))),
        "injury_claimed": float(bool(ctx.get("injury_claimed"))),
        "prior_claims_12m": float(ctx.get("prior_claims_12m") or 0),
        "customer_tenure_months": float(ctx.get("customer_tenure_months") or 0),
        "vehicle_age": float(ctx.get("vehicle_age") or 0),
        "vehicle_power": float(ctx.get("vehicle_power") or 0),
        "bonus_malus": float(ctx.get("bonus_malus") or 50),
        "description_word_count": float(len(words)),
        "description_is_generic": float(
            any(p in description.lower() for p in _GENERIC_PHRASES) or len(words) < 6
        ),
        "document_count": float(ctx.get("document_count") or 0),
        "missing_invoice": float(not bool(ctx.get("has_invoice"))),
        "vendor_watchlisted": float(bool(ctx.get("vendor_watchlisted"))),
        "vendor_claims_serviced": float(ctx.get("vendor_claims_serviced") or 0),
        "vendor_invoice_ratio": _safe_ratio(
            claimed, float(ctx.get("vendor_avg_invoice") or claimed or 1.0), cap=10.0
        ),
        "incident_type_code": float(
            INCIDENT_TYPES.index(ctx["incident_type"])
            if ctx.get("incident_type") in INCIDENT_TYPES
            else len(INCIDENT_TYPES)
        ),
        # graph features - zero when the graph has not been built yet
        "ring_size": float(ctx.get("ring_size") or 0),
        "ring_cohesion": float(ctx.get("ring_cohesion") or 0.0),
        "shared_phone_degree": float(ctx.get("shared_phone_degree") or 0),
        "shared_account_degree": float(ctx.get("shared_account_degree") or 0),
        "vendor_degree": float(ctx.get("vendor_degree") or 0),
        "neighbour_mean_amount_ratio": _safe_ratio(
            claimed, float(ctx.get("neighbour_mean_amount") or claimed or 1.0), cap=10.0
        ),
    }
    return row


def to_vector(row: dict[str, float], feature_names: list[str] | None = None) -> list[float]:
    names = feature_names or FEATURE_NAMES
    return [float(row.get(n, 0.0)) for n in names]

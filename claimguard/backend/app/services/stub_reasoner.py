"""Offline rule-based reasoners.

These stand in for Gemini when no API key is configured. They are deliberately
*real* logic rather than canned strings: given the same claim they produce the
same verdict every time, which makes the pipeline demo-able, unit-testable and
reproducible. Set GEMINI_API_KEY and the same prompts go to the real model.
"""

from __future__ import annotations

from typing import Any, Callable

Handler = Callable[[dict[str, Any]], dict[str, Any]]


def _f(ctx: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(ctx.get(key) or default)
    except (TypeError, ValueError):
        return default


def _band(score: float, bands: list[tuple[float, str]]) -> str:
    for threshold, label in bands:
        if score <= threshold:
            return label
    return bands[-1][1]


# --------------------------------------------------------------------- intake
def intake_normalisation(ctx: dict[str, Any]) -> dict[str, Any]:
    claim = ctx.get("claim", {})
    required = [
        "incident_type",
        "incident_date",
        "incident_description",
        "claimed_amount",
        "incident_postal_code",
    ]
    missing = [f for f in required if not claim.get(f)]
    description = (claim.get("incident_description") or "").lower()
    amount = _f(claim, "claimed_amount")

    early_flags: list[str] = []
    if claim.get("reporting_delay_days", 0) and float(claim["reporting_delay_days"]) > 14:
        early_flags.append(
            f"Reported {int(float(claim['reporting_delay_days']))} days after the incident"
        )
    if claim.get("days_since_policy_start") is not None and float(
        claim["days_since_policy_start"]
    ) < 30:
        early_flags.append("Incident occurred within 30 days of policy inception")
    if not claim.get("police_report") and amount > 10000:
        early_flags.append("High-value claim filed without a police report")
    if not claim.get("witnesses"):
        early_flags.append("No witnesses recorded")
    if len(description.split()) < 8:
        early_flags.append("Incident description is unusually short and low in detail")

    completeness = round(max(0.0, 1.0 - 0.15 * len(missing) - 0.05 * len(early_flags)), 3)
    severity = _band(
        amount, [(2500, "minor"), (8000, "moderate"), (20000, "major"), (float("inf"), "severe")]
    )
    return {
        "normalized_incident_type": claim.get("incident_type", "collision"),
        "severity_band": severity,
        "completeness_score": completeness,
        "missing_fields": missing,
        "narrative_summary": (
            f"{str(claim.get('incident_type', 'collision')).replace('_', ' ').title()} claim for "
            f"{amount:,.0f} filed by {claim.get('customer_name', 'the policyholder')} "
            f"in {claim.get('incident_postal_code', 'an unspecified area')}."
        ),
        "early_flags": early_flags,
    }


# ------------------------------------------------------------------ documents
def document_consistency(ctx: dict[str, Any]) -> dict[str, Any]:
    claim = ctx.get("claim", {})
    docs = ctx.get("documents", [])
    mismatches: list[dict[str, Any]] = []
    claimed = _f(claim, "claimed_amount")

    invoice_totals: list[float] = []
    seen_invoice_numbers: dict[str, int] = {}

    for doc in docs:
        fields = doc.get("extracted_fields") or {}
        total = fields.get("total_amount")
        if total is not None:
            invoice_totals.append(float(total))
            if claimed and abs(float(total) - claimed) / max(claimed, 1.0) > 0.15:
                mismatches.append(
                    {
                        "field": "total_amount",
                        "document": doc.get("filename"),
                        "claim_value": claimed,
                        "document_value": float(total),
                        "severity": "high"
                        if abs(float(total) - claimed) / max(claimed, 1.0) > 0.4
                        else "medium",
                    }
                )
        inv_no = fields.get("invoice_number")
        if inv_no:
            seen_invoice_numbers[str(inv_no)] = seen_invoice_numbers.get(str(inv_no), 0) + 1
        doc_date = fields.get("service_date")
        if doc_date and claim.get("incident_date") and str(doc_date) < str(claim["incident_date"])[:10]:
            mismatches.append(
                {
                    "field": "service_date",
                    "document": doc.get("filename"),
                    "claim_value": str(claim["incident_date"])[:10],
                    "document_value": str(doc_date),
                    "severity": "high",
                }
            )
        plate = fields.get("registration")
        if plate and claim.get("vehicle_registration") and plate != claim["vehicle_registration"]:
            mismatches.append(
                {
                    "field": "registration",
                    "document": doc.get("filename"),
                    "claim_value": claim["vehicle_registration"],
                    "document_value": plate,
                    "severity": "high",
                }
            )

    duplicates = [k for k, v in seen_invoice_numbers.items() if v > 1]
    penalty = sum(0.25 if m["severity"] == "high" else 0.12 for m in mismatches)
    penalty += 0.3 * len(duplicates)
    if not docs:
        penalty += 0.2
    consistency = round(max(0.0, 1.0 - penalty), 3)

    return {
        "consistency_score": consistency,
        "documents_reviewed": len(docs),
        "mismatches": mismatches,
        "duplicate_invoice_numbers": duplicates,
        "invoice_total": round(sum(invoice_totals), 2) if invoice_totals else None,
        "notes": (
            "No supporting documents were supplied, so amounts could not be corroborated."
            if not docs
            else f"Reviewed {len(docs)} document(s); found {len(mismatches)} field mismatch(es)."
        ),
    }


# --------------------------------------------------------------------- damage
def damage_assessment(ctx: dict[str, Any]) -> dict[str, Any]:
    claim = ctx.get("claim", {})
    findings = ctx.get("image_findings", [])
    claimed = _f(claim, "claimed_amount")
    estimate = _f(claim, "estimated_repair_cost") or claimed

    severity_scores = {"minor": 1, "moderate": 2, "severe": 3}
    detected = [f.get("severity", "minor") for f in findings]
    worst = max((severity_scores.get(s, 1) for s in detected), default=0)
    label = {0: "none_detected", 1: "minor", 2: "moderate", 3: "severe"}[worst]

    expected_ranges = {
        "none_detected": (0, 1500),
        "minor": (300, 3500),
        "moderate": (2500, 12000),
        "severe": (9000, 45000),
    }
    lo, hi = expected_ranges[label]
    inconsistencies: list[str] = []
    if findings and claimed > hi * 1.2:
        # Only meaningful when there are photographs to compare against; with no
        # images the cost range says nothing, so it is not held against the claim.
        inconsistencies.append(
            f"Claimed amount {claimed:,.0f} exceeds the expected range for {label} damage "
            f"({lo:,.0f}-{hi:,.0f})"
        )
    if estimate and claimed and claimed > estimate * 1.3:
        inconsistencies.append("Claimed amount is more than 30% above the repair estimate")

    plausibility = 1.0 - 0.3 * len(inconsistencies)
    if not findings:
        plausibility -= 0.1  # unverified, not implausible
    plausibility = round(max(0.0, min(1.0, plausibility)), 3)

    return {
        "damage_severity": label,
        "regions_detected": [f.get("region") for f in findings if f.get("region")],
        "plausibility_score": plausibility,
        "expected_cost_range": [lo, hi],
        "inconsistencies": inconsistencies,
        "notes": (
            f"Vision module reported {len(findings)} damage region(s); overall severity "
            f"assessed as {label}."
        ),
    }


# --------------------------------------------------------------------- policy
def policy_coverage_review(ctx: dict[str, Any]) -> dict[str, Any]:
    claim = ctx.get("claim", {})
    policy = ctx.get("policy", {})
    clauses = ctx.get("retrieved_clauses", [])

    claimed = _f(claim, "claimed_amount")
    limit = _f(policy, "coverage_limit", 25000.0)
    deductible = _f(policy, "deductible", 500.0)
    exclusions = [str(e).lower() for e in (policy.get("exclusions") or [])]
    incident = str(claim.get("incident_type", "")).lower()
    description = str(claim.get("incident_description", "")).lower()

    triggered: list[dict[str, Any]] = []
    for exclusion in exclusions:
        token = exclusion.replace("_", " ")
        if token in incident or token in description:
            triggered.append({"exclusion": exclusion, "matched_on": token})

    policy_active = policy.get("status", "active") == "active"
    within_period = bool(claim.get("within_policy_period", True))

    covered = policy_active and within_period and not triggered
    payable = 0.0
    if covered:
        payable = max(0.0, min(claimed, limit) - deductible)

    confidence = 0.6 + 0.1 * min(len(clauses), 4)
    if not clauses:
        confidence = 0.45

    reasoning_bits = []
    if not policy_active:
        reasoning_bits.append(f"Policy status is '{policy.get('status')}', not active.")
    if not within_period:
        reasoning_bits.append("Incident date falls outside the policy period.")
    for t in triggered:
        reasoning_bits.append(f"Exclusion '{t['exclusion']}' is triggered by the incident details.")
    if covered:
        reasoning_bits.append(
            f"Loss is covered under the comprehensive motor section; payable is the claimed "
            f"amount capped at the {limit:,.0f} limit, less the {deductible:,.0f} deductible."
        )

    return {
        "covered": covered,
        "coverage_confidence": round(min(confidence, 0.95), 3),
        "applicable_clauses": [
            {"clause_id": c.get("clause_id"), "title": c.get("title"), "score": c.get("score")}
            for c in clauses[:4]
        ],
        "exclusions_triggered": triggered,
        "deductible_applied": deductible,
        "coverage_limit": limit,
        "payable_estimate": round(payable, 2),
        "reasoning": " ".join(reasoning_bits) or "Coverage assessed against the retrieved clauses.",
    }


# -------------------------------------------------------------------- anomaly
def anomaly_explanation(ctx: dict[str, Any]) -> dict[str, Any]:
    prob = _f(ctx, "fraud_probability")
    features = ctx.get("top_features", [])
    drivers = []
    for feat in features[:5]:
        direction = "raises" if float(feat.get("contribution", 0)) >= 0 else "lowers"
        drivers.append(
            {
                "feature": feat.get("name"),
                "value": feat.get("value"),
                "effect": direction,
                "weight": round(abs(float(feat.get("contribution", 0))), 4),
            }
        )
    severity = _band(prob, [(0.25, "low"), (0.55, "moderate"), (0.8, "high"), (1.01, "critical")])
    top = ", ".join(str(d["feature"]) for d in drivers[:3]) or "no dominant feature"
    return {
        "severity": severity,
        "drivers": drivers,
        "explanation": (
            f"The gradient-boosted anomaly model scores this claim at {prob:.0%} fraud "
            f"probability ({severity} risk). The score is driven mainly by {top}."
        ),
    }


# -------------------------------------------------------------------- network
def network_assessment(ctx: dict[str, Any]) -> dict[str, Any]:
    ring = ctx.get("ring") or {}
    shared = ring.get("shared_attributes", [])
    neighbours = ctx.get("neighbour_claims", [])
    size = int(ring.get("size", 0) or 0)
    cohesion = _f(ring, "cohesion")

    if size >= 4 and cohesion > 0.5:
        risk = "high"
    elif size >= 3:
        risk = "moderate"
    elif size == 2:
        risk = "low"
    else:
        risk = "none"

    suspicious_links = []
    for n in neighbours[:8]:
        suspicious_links.append(
            {
                "claim_number": n.get("claim_number"),
                "relationship": n.get("relationship"),
                "shared_value": n.get("shared_value"),
                "claimed_amount": n.get("claimed_amount"),
            }
        )

    narrative = (
        "No connected component of suspicious claims was found around this claim."
        if risk == "none"
        else (
            f"This claim sits in a cluster of {size} claims linked by "
            f"{', '.join(shared) or 'shared identifiers'} (cohesion {cohesion:.2f}). "
            f"Network risk assessed as {risk}."
        )
    )
    return {
        "ring_risk": risk,
        "ring_size": size,
        "shared_attributes": shared,
        "cohesion": round(cohesion, 3),
        "suspicious_links": suspicious_links,
        "narrative": narrative,
        "recommended_expansion": [n.get("claim_number") for n in neighbours[:3]],
    }


# ---------------------------------------------------------------- adjudication
def final_adjudication(ctx: dict[str, Any]) -> dict[str, Any]:
    score = _f(ctx, "fraud_score")
    coverage = ctx.get("coverage", {})
    signals = ctx.get("risk_signals", [])
    payable = _f(coverage, "payable_estimate")
    auto_approve_below = _f(ctx, "auto_approve_below", 0.25)
    auto_reject_above = _f(ctx, "auto_reject_above", 0.85)

    if not coverage.get("covered", True):
        decision = "reject"
        rationale = (
            "The loss is not payable under the policy: "
            + str(coverage.get("reasoning", "coverage conditions were not met."))
        )
        payout = 0.0
    elif score >= auto_reject_above:
        decision = "reject"
        rationale = (
            f"Fraud score of {score:.0%} is above the automatic rejection threshold and is "
            "supported by corroborating signals across document, damage and network checks."
        )
        payout = 0.0
    elif score <= auto_approve_below:
        decision = "approve"
        rationale = (
            f"Fraud score of {score:.0%} is low, the claim is covered, and no material "
            "inconsistencies were found in the supporting evidence."
        )
        payout = payable
    else:
        decision = "review"
        rationale = (
            f"Fraud score of {score:.0%} falls in the manual-review band. "
            f"{len(signals)} risk signal(s) need an investigator's judgement before payment."
        )
        payout = payable

    distance = min(abs(score - auto_approve_below), abs(score - auto_reject_above))
    confidence = round(min(0.99, 0.55 + distance + 0.03 * len(signals)), 3)

    next_actions = {
        "approve": ["Release payment", "Archive evidence pack"],
        "review": [
            "Assign to a fraud investigator",
            "Request original invoices from the garage",
            "Verify the incident with the reporting police station",
        ],
        "reject": [
            "Issue a written decline with the clause reference",
            "Refer the linked claims to the SIU",
        ],
    }[decision]

    return {
        "decision": decision,
        "confidence": confidence,
        "rationale": rationale,
        "recommended_payout": round(payout, 2),
        "key_evidence": [s.get("detail") for s in signals[:5] if isinstance(s, dict)],
        "next_actions": next_actions,
    }


STUB_HANDLERS: dict[str, Handler] = {
    "intake_normalisation": intake_normalisation,
    "document_consistency": document_consistency,
    "damage_assessment": damage_assessment,
    "policy_coverage_review": policy_coverage_review,
    "anomaly_explanation": anomaly_explanation,
    "network_assessment": network_assessment,
    "final_adjudication": final_adjudication,
}

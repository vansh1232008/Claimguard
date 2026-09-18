"""Evaluation and ablation.

    python ml/evaluate.py --data data/claims_dataset.csv

Trains the same model three ways on the same time-based split and prints one
table, so the contribution of each layer is measurable instead of asserted:

    1. base features only          (claim-level signals)
    2. + hand-made graph features  (ring size, cohesion, shared-identity degree)
    3. + GraphSAGE embeddings      (if artifacts/graphsage_embeddings.npz exists)

It also reports ring-level capture: of the fraud rings in the held-out period,
how many had at least one member flagged. That is the number that matters
operationally — catching one member of a ring opens the whole cluster.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from train_xgboost import build_matrix, load_embeddings, metrics_at, pick_threshold, time_split
from graph_features import read_rows


def train_once(X, y, train_idx, val_idx, test_idx, feature_names, rounds=300):
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    pos = int(y[train_idx].sum())
    neg = len(train_idx) - pos
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=feature_names)
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx], feature_names=feature_names)
    dtest = xgb.DMatrix(X[test_idx], label=y[test_idx], feature_names=feature_names)

    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "gamma": 0.5,
            "scale_pos_weight": neg / max(pos, 1),
            "tree_method": "hist",
            "seed": 42,
        },
        dtrain,
        num_boost_round=rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )
    val_probs = booster.predict(dval)
    # F1-optimal on validation. Chasing a fixed 90% recall at an 8% base rate
    # buys recall with precision so low the referral queue is unworkable, so the
    # ablation is compared at the balanced point instead.
    threshold = pick_threshold(y[val_idx], val_probs, None)
    test_probs = booster.predict(dtest)
    return {
        "roc_auc": round(float(roc_auc_score(y[test_idx], test_probs)), 4),
        "pr_auc": round(float(average_precision_score(y[test_idx], test_probs)), 4),
        **metrics_at(y[test_idx], test_probs, threshold),
    }, test_probs, threshold


def ring_capture(rows, test_idx, probs, threshold) -> dict:
    rings: dict[str, list[bool]] = defaultdict(list)
    for pos, idx in enumerate(test_idx):
        row = rows[int(idx)]
        ring = row.get("fraud_ring_id")
        if ring:
            rings[ring].append(bool(probs[pos] >= threshold))
    if not rings:
        return {"rings_in_test": 0, "rings_with_a_hit": 0, "ring_capture_rate": 0.0}
    hits = sum(1 for flags in rings.values() if any(flags))
    member_recall = sum(sum(f) for f in rings.values()) / sum(len(f) for f in rings.values())
    return {
        "rings_in_test": len(rings),
        "rings_with_a_hit": hits,
        "ring_capture_rate": round(hits / len(rings), 4),
        "ring_member_recall": round(member_recall, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/claims_dataset.csv")
    parser.add_argument("--embeddings", default="artifacts/graphsage_embeddings.npz")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _bootstrap.REPO_ROOT / data_path
    if not data_path.exists():
        raise SystemExit(f"{data_path} not found. Run ml/generate_dataset.py first.")

    rows = read_rows(data_path)
    print(f"Loaded {len(rows):,} claims\n")

    configs = []

    X, y, order, names = build_matrix(rows, use_graph=False)
    train, val, test = time_split(order)
    configs.append(("base features", X, names))

    Xg, _, _, names_g = build_matrix(rows, use_graph=True)
    configs.append(("+ graph features", Xg, names_g))

    emb_path = Path(args.embeddings)
    if not emb_path.is_absolute():
        emb_path = _bootstrap.REPO_ROOT / emb_path
    if emb_path.exists():
        embeddings, dim = load_embeddings(emb_path)
        Xe, _, _, names_e = build_matrix(rows, use_graph=True, embeddings=embeddings, emb_dim=dim)
        configs.append(("+ GraphSAGE embeddings", Xe, names_e))
    else:
        print(f"(skipping GraphSAGE ablation - {emb_path} not found)\n")

    results = {}
    last = None
    for label, matrix, feature_names in configs:
        print(f"Training: {label} ({matrix.shape[1]} features) ...")
        metrics, probs, threshold = train_once(matrix, y, train, val, test, feature_names)
        metrics.update(ring_capture(rows, test, probs, threshold))
        results[label] = metrics
        last = metrics

    print("\n" + "=" * 92)
    header = f"{'configuration':<26}{'ROC-AUC':>9}{'PR-AUC':>9}{'recall':>9}{'precision':>11}{'F1':>8}{'ring capture':>15}"
    print(header)
    print("-" * 92)
    for label, m in results.items():
        print(
            f"{label:<26}{m['roc_auc']:>9.4f}{m['pr_auc']:>9.4f}{m['recall']:>9.3f}"
            f"{m['precision']:>11.3f}{m['f1']:>8.3f}{m.get('ring_capture_rate', 0):>15.3f}"
        )
    print("=" * 92)

    out = _bootstrap.ARTIFACT_DIR / "evaluation_report.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    best_label, best = max(results.items(), key=lambda kv: kv[1]["pr_auc"])
    baseline = results["base features"]
    print(f"\nBest configuration: {best_label}")
    print(
        f"  ROC-AUC {best['roc_auc']:.3f} | PR-AUC {best['pr_auc']:.3f} "
        f"(base features: {baseline['pr_auc']:.3f}) on a held-out "
        f"{len(test):,}-claim test period at an {float(y.mean()):.1%} fraud rate."
    )
    print(
        f"  At the balanced operating point: recall {best['recall']:.0%}, "
        f"precision {best['precision']:.0%}."
    )
    print(
        f"  Ring capture: {best.get('ring_capture_rate', 0):.0%} of the fraud rings in the "
        f"test period had at least one member flagged (base features: "
        f"{baseline.get('ring_capture_rate', 0):.0%})."
    )
    print(
        "\nQuote these numbers, not the ones in any earlier draft of the CV - they are the "
        "ones this repo reproduces."
    )


if __name__ == "__main__":
    main()

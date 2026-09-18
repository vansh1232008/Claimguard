"""Train the supervised fraud model.

    python ml/train_xgboost.py --data data/claims_dataset.csv

Notes that matter for the interview:

* Split is **time-based**, not random. Claims are ordered by incident date and
  the last 20% is held out. A random split leaks ring structure across the
  split — members of the same ring end up on both sides — and the model looks
  far better than it is.
* The positive class is ~8%, so ``scale_pos_weight`` is set from the training
  fold and the threshold is chosen on the validation fold for recall, not left
  at 0.5.
* Graph features are optional (``--no-graph``) so the lift they give can be
  measured rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from app.services.features import BASE_FEATURES, FEATURE_NAMES, build_feature_row, to_vector
from graph_features import build_graph, graph_feature_table, read_rows, row_to_context


def load_embeddings(path: Path | None):
    if path is None:
        return {}, 0
    if not path.is_absolute():
        path = _bootstrap.REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"{path} not found. Run ml/train_graphsage.py first.")
    blob = np.load(path, allow_pickle=True)
    ids = [str(i) for i in blob["claim_ids"]]
    vecs = blob["embeddings"]
    print(f"Loaded {len(ids):,} GraphSAGE embeddings (dim {vecs.shape[1]})")
    return {ids[i]: vecs[i] for i in range(len(ids))}, int(vecs.shape[1])


def build_matrix(rows, use_graph: bool, embeddings: dict | None = None, emb_dim: int = 0):
    feature_names = list(FEATURE_NAMES if use_graph else BASE_FEATURES)
    if emb_dim:
        feature_names += [f"sage_{i}" for i in range(emb_dim)]
    graph_table: dict[str, dict[str, float]] = {}
    if use_graph:
        print("Building relationship graph ...")
        t0 = time.perf_counter()
        store = build_graph(rows)
        graph_table = graph_feature_table(store)
        print(
            f"  {store.graph.number_of_nodes():,} nodes / "
            f"{store.graph.number_of_edges():,} edges in {time.perf_counter() - t0:.1f}s"
        )

    X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int8)
    order = np.zeros(len(rows), dtype="datetime64[s]")

    base_names = feature_names[: len(feature_names) - emb_dim] if emb_dim else feature_names
    for i, row in enumerate(rows):
        ctx = row_to_context(row, graph_table.get(row["claim_id"]))
        vec = to_vector(build_feature_row(ctx), base_names)
        if emb_dim:
            emb = (embeddings or {}).get(row["claim_id"])
            vec = vec + (list(emb) if emb is not None else [0.0] * emb_dim)
        X[i] = vec
        y[i] = int(float(row.get("is_fraud") or 0))
        order[i] = np.datetime64(str(row["incident_date"])[:19])
    return X, y, order, feature_names


def time_split(order: np.ndarray, test_size: float = 0.2, val_size: float = 0.1):
    idx = np.argsort(order)
    n = len(idx)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    train = idx[: n - n_test - n_val]
    val = idx[n - n_test - n_val : n - n_test]
    test = idx[n - n_test :]
    return train, val, test


def metrics_at(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def pick_threshold(y_val: np.ndarray, probs: np.ndarray, target_recall: float | None) -> float:
    """Choose the operating point on the validation fold, never on test.

    With ``target_recall`` set, take the highest threshold that still reaches
    it (highest precision for that recall). With it unset, maximise F1 — the
    sensible default when nobody has stated the cost of a missed fraud versus
    the cost of a wasted investigation.
    """
    if target_recall:
        best = 0.05
        for t in np.arange(0.05, 0.95, 0.01):
            if metrics_at(y_val, probs, float(t))["recall"] >= target_recall:
                best = float(t)
        return best
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        m = metrics_at(y_val, probs, float(t))
        if m["f1"] > best_f1:
            best_t, best_f1 = float(t), m["f1"]
    return best_t


def fit_calibration(y_val: np.ndarray, probs: np.ndarray) -> dict:
    """Platt scaling on the validation fold.

    ``scale_pos_weight`` is set to ~11 so the model learns from an 8%-positive
    class, but that makes its raw output a *ranking* score, not a probability:
    ordinary claims come back at 0.7+. Everything downstream treats the number
    as a probability - the approve/refer thresholds, the blend with the rule
    signals, the number shown to the investigator - so it has to be calibrated
    before it leaves the model.

    A logistic regression on the validation log-odds does it in two parameters,
    which stay readable in the artifact file. Ranking metrics (ROC-AUC, PR-AUC)
    are unchanged by a monotonic transform; only the thresholds move.
    """
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    logits = np.log(np.clip(probs, eps, 1 - eps) / (1 - np.clip(probs, eps, 1 - eps)))
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logits.reshape(-1, 1), y_val)
    return {
        "method": "platt",
        "a": float(lr.coef_[0][0]),
        "b": float(lr.intercept_[0]),
    }


def apply_calibration(probs: np.ndarray, calibration: dict) -> np.ndarray:
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    z = calibration["a"] * logits + calibration["b"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def operating_points(y_true: np.ndarray, probs: np.ndarray) -> list[dict]:
    """A small table of alternatives, so the threshold is a business choice."""
    points = []
    for label, target in (("recall 70%", 0.70), ("recall 80%", 0.80), ("recall 90%", 0.90)):
        chosen = 0.05
        for t in np.arange(0.05, 0.95, 0.01):
            if metrics_at(y_true, probs, float(t))["recall"] >= target:
                chosen = float(t)
        points.append({"target": label, **metrics_at(y_true, probs, chosen)})
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/claims_dataset.csv")
    parser.add_argument("--no-graph", action="store_true", help="train on base features only")
    parser.add_argument("--embeddings", default=None, help="path to graphsage_embeddings.npz")
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--target-recall", type=float, default=None,
                        help="fix the operating point at this recall; default maximises F1")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _bootstrap.REPO_ROOT / data_path
    if not data_path.exists():
        raise SystemExit(
            f"{data_path} not found. Run: python ml/generate_dataset.py --claims 150000"
        )

    print(f"Loading {data_path} ...")
    rows = read_rows(data_path)
    print(f"  {len(rows):,} claims, {sum(int(float(r['is_fraud'])) for r in rows):,} fraudulent")

    embeddings, emb_dim = load_embeddings(Path(args.embeddings) if args.embeddings else None)
    X, y, order, feature_names = build_matrix(
        rows, use_graph=not args.no_graph, embeddings=embeddings, emb_dim=emb_dim
    )
    train, val, test = time_split(order)
    print(f"Split (time-based): train {len(train):,} | val {len(val):,} | test {len(test):,}")

    pos = int(y[train].sum())
    neg = len(train) - pos
    scale_pos_weight = neg / max(pos, 1)

    dtrain = xgb.DMatrix(X[train], label=y[train], feature_names=feature_names)
    dval = xgb.DMatrix(X[val], label=y[val], feature_names=feature_names)
    dtest = xgb.DMatrix(X[test], label=y[test], feature_names=feature_names)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["aucpr", "auc"],
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 4,
        "gamma": 0.5,
        "lambda": 1.5,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "seed": 42,
    }
    print(f"Training (scale_pos_weight={scale_pos_weight:.2f}) ...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=40,
        verbose_eval=50,
    )

    raw_val_probs = booster.predict(dval)
    calibration = fit_calibration(y[val], raw_val_probs)
    val_probs = apply_calibration(raw_val_probs, calibration)
    threshold = pick_threshold(y[val], val_probs, args.target_recall)
    test_probs = apply_calibration(booster.predict(dtest), calibration)

    mean_pred = float(test_probs.mean())
    print(
        f"\nCalibration: a={calibration['a']:.3f} b={calibration['b']:.3f} | "
        f"mean predicted {mean_pred:.3f} vs actual fraud rate {float(y[test].mean()):.3f}"
    )

    report = {
        "model": "xgboost",
        "features": feature_names,
        "graph_features_used": not args.no_graph,
        "best_iteration": int(booster.best_iteration),
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y[test], test_probs)), 4),
        "pr_auc": round(float(average_precision_score(y[test], test_probs)), 4),
        "at_threshold": metrics_at(y[test], test_probs, threshold),
        "at_0.5": metrics_at(y[test], test_probs, 0.5),
        "operating_points": operating_points(y[test], test_probs),
        "calibration": calibration,
        "mean_predicted_probability": round(mean_pred, 4),
        "actual_fraud_rate_test": round(float(y[test].mean()), 4),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "fraud_rate": round(float(y.mean()), 4),
    }

    art = _bootstrap.ARTIFACT_DIR
    model_file = Path(args.out) if args.out else art / "fraud_xgboost.json"
    booster.save_model(str(model_file))
    (art / "feature_metadata.json").write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "threshold": threshold,
                "calibration": calibration,
            },
            indent=2,
        )
    )
    (art / "xgboost_report.json").write_text(json.dumps(report, indent=2))

    gains = booster.get_score(importance_type="gain")
    top = sorted(gains.items(), key=lambda kv: -kv[1])[:12]

    print("\n=== Test metrics ===")
    print(f"ROC-AUC      {report['roc_auc']}")
    print(f"PR-AUC       {report['pr_auc']}")
    t = report["at_threshold"]
    print(f"@{t['threshold']:.2f}  recall {t['recall']:.3f}  precision {t['precision']:.3f}  f1 {t['f1']:.3f}")
    print("\nOperating points (test fold):")
    print(f"  {'target':<12}{'threshold':>10}{'recall':>9}{'precision':>11}{'F1':>8}")
    for p in report["operating_points"]:
        print(
            f"  {p['target']:<12}{p['threshold']:>10.2f}{p['recall']:>9.3f}"
            f"{p['precision']:>11.3f}{p['f1']:>8.3f}"
        )
    print("\nTop features by gain:")
    for name, gain in top:
        print(f"  {name:<32} {gain:,.1f}")
    print(f"\nSaved model -> {model_file}")
    print(f"Saved report -> {art / 'xgboost_report.json'}")


if __name__ == "__main__":
    main()

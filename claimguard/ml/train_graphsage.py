"""Unsupervised GraphSAGE over the claim relationship graph.

    python ml/train_graphsage.py --data data/claims_dataset.csv --dim 16

What it does: learns a vector per claim such that claims connected in the
relationship graph land close together and unconnected claims land apart
(the standard unsupervised GraphSAGE objective — link prediction with negative
sampling). Those vectors are then fed to the XGBoost model as extra columns:

    python ml/train_xgboost.py --embeddings artifacts/graphsage_embeddings.npz

Why it helps: the hand-made graph features (ring size, cohesion, degrees)
summarise a neighbourhood with a handful of numbers. The embedding keeps the
*shape* of the neighbourhood — a claim two hops from a dense identity-sharing
cluster looks different from one two hops from a busy garage, even when both
have the same degree.

Implementation: two mean-aggregator layers written in NumPy, trained with
mini-batch SGD. Written without a deep-learning framework on purpose so the
project has no heavyweight install; if PyTorch Geometric is available the same
objective can be swapped in one-for-one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from graph_features import build_graph, read_rows, row_to_context
from app.services.features import BASE_FEATURES, build_feature_row, to_vector


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class MeanAggregatorSAGE:
    """Two-layer GraphSAGE with mean aggregation, NumPy only."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / (in_dim * 2))
        scale2 = np.sqrt(2.0 / (hidden * 2))
        # Each layer concatenates [self, mean(neighbours)] then projects.
        self.W1 = rng.normal(0, scale1, size=(in_dim * 2, hidden))
        self.W2 = rng.normal(0, scale2, size=(hidden * 2, out_dim))

    @staticmethod
    def _neighbour_mean(H: np.ndarray, adj: list[np.ndarray], nodes: np.ndarray) -> np.ndarray:
        out = np.zeros((len(nodes), H.shape[1]), dtype=np.float32)
        for i, n in enumerate(nodes):
            nbrs = adj[n]
            if len(nbrs):
                out[i] = H[nbrs].mean(axis=0)
        return out

    def layer1(self, X: np.ndarray, adj: list[np.ndarray]) -> np.ndarray:
        agg1 = self._neighbour_mean(X, adj, np.arange(len(X)))
        return np.maximum(0.0, np.concatenate([X, agg1], axis=1) @ self.W1)

    def layer2_input(self, h1: np.ndarray, adj: list[np.ndarray], nodes: np.ndarray) -> np.ndarray:
        agg2 = self._neighbour_mean(h1, adj, nodes)
        return np.concatenate([h1[nodes], agg2], axis=1)

    def forward(self, h1: np.ndarray, adj: list[np.ndarray], nodes: np.ndarray):
        """Unnormalised embeddings. Normalisation happens at embed time only.

        Training on raw dot products keeps the gradient of the loss with
        respect to W2 exact; normalising inside the forward pass and ignoring
        its Jacobian is what stopped the loss descending in the first version.
        """
        feats = self.layer2_input(h1, adj, nodes)
        return feats @ self.W2, feats

    def embed_all(self, X: np.ndarray, adj: list[np.ndarray]) -> np.ndarray:
        h1 = self.layer1(X, adj)
        z, _ = self.forward(h1, adj, np.arange(len(X)))
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return z / norms


def train(
    X: np.ndarray,
    adj: list[np.ndarray],
    edges: np.ndarray,
    dim: int = 16,
    hidden: int = 32,
    epochs: int = 5,
    batch: int = 1024,
    lr: float = 0.02,
    neg_samples: int = 3,
    seed: int = 42,
) -> tuple[MeanAggregatorSAGE, list[float]]:
    rng = np.random.default_rng(seed)
    model = MeanAggregatorSAGE(X.shape[1], hidden, dim, seed=seed)
    n_nodes = len(X)
    losses: list[float] = []

    # Layer 1 is a fixed random projection (its weights are not trained), so its
    # output can be computed once instead of on every batch. That is the single
    # biggest speed-up in this trainer.
    h1 = model.layer1(X, adj)

    for epoch in range(epochs):
        rng.shuffle(edges)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(edges), batch):
            chunk = edges[start : start + batch]
            if len(chunk) == 0:
                continue
            src, dst = chunk[:, 0], chunk[:, 1]
            neg = rng.integers(0, n_nodes, size=(len(chunk), neg_samples))

            feats_s = model.layer2_input(h1, adj, src)
            feats_d = model.layer2_input(h1, adj, dst)
            feats_n = model.layer2_input(h1, adj, neg.ravel())

            zs = feats_s @ model.W2
            zd = feats_d @ model.W2
            zn = (feats_n @ model.W2).reshape(len(chunk), neg_samples, -1)

            pos_score = np.sum(zs * zd, axis=1)
            neg_score = np.einsum("ij,ikj->ik", zs, zn)

            pos_loss = -np.log(sigmoid(pos_score) + 1e-9)
            neg_loss = -np.log(1 - sigmoid(neg_score) + 1e-9).sum(axis=1)
            loss = float(np.mean(pos_loss + neg_loss))
            epoch_loss += loss
            n_batches += 1

            # dL/dz for each participating embedding.
            d_pos = (sigmoid(pos_score) - 1.0)[:, None]      # (B, 1)
            d_neg = sigmoid(neg_score)[:, :, None]           # (B, K, 1)

            g_zs = d_pos * zd + np.einsum("ikj,ikl->il", d_neg, zn)
            g_zd = d_pos * zs
            g_zn = (d_neg * zs[:, None, :]).reshape(-1, zs.shape[1])

            grad_W2 = (
                feats_s.T @ g_zs + feats_d.T @ g_zd + feats_n.T @ g_zn
            ) / len(chunk)
            model.W2 -= lr * np.clip(grad_W2, -5.0, 5.0)

        mean_loss = epoch_loss / max(n_batches, 1)
        losses.append(round(mean_loss, 5))
        print(f"  epoch {epoch + 1}/{epochs}  loss {mean_loss:.4f}")
    return model, losses


def link_prediction_auc(Z: np.ndarray, edges: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    sample = edges[rng.choice(len(edges), size=min(5000, len(edges)), replace=False)]
    pos = np.sum(Z[sample[:, 0]] * Z[sample[:, 1]], axis=1)
    neg_src = rng.integers(0, len(Z), size=len(sample))
    neg_dst = rng.integers(0, len(Z), size=len(sample))
    neg = np.sum(Z[neg_src] * Z[neg_dst], axis=1)
    wins = (pos[:, None] > neg[None, :]).mean()
    return float(wins)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/claims_dataset.csv")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=60000,
                        help="cap the graph size for the NumPy trainer")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _bootstrap.REPO_ROOT / data_path
    if not data_path.exists():
        raise SystemExit(f"{data_path} not found. Run ml/generate_dataset.py first.")

    rows = read_rows(data_path)
    print(f"Loaded {len(rows):,} claims")

    store = build_graph(rows)
    graph = store.graph
    print(f"Graph: {graph.number_of_nodes():,} nodes / {graph.number_of_edges():,} edges")

    # Keep only connected claims: isolated nodes carry no graph information and
    # would dominate the node set on a sparse graph.
    connected = [n for n in graph.nodes if graph.degree(n) > 0][: args.max_nodes]
    index = {n: i for i, n in enumerate(connected)}
    print(f"Training on {len(connected):,} connected claims")

    by_id = {r["claim_id"]: r for r in rows}
    X = np.zeros((len(connected), len(BASE_FEATURES)), dtype=np.float32)
    for node, i in index.items():
        X[i] = to_vector(build_feature_row(row_to_context(by_id[node])), BASE_FEATURES)
    # standardise so the random projection behaves
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    adj: list[np.ndarray] = []
    for node in connected:
        nbrs = [index[nb] for nb in graph.neighbors(node) if nb in index]
        adj.append(np.array(nbrs, dtype=np.int64))

    edges = np.array(
        [[index[u], index[v]] for u, v in graph.edges if u in index and v in index],
        dtype=np.int64,
    )
    if len(edges) == 0:
        raise SystemExit("Graph has no edges - nothing to train on.")

    t0 = time.perf_counter()
    model, losses = train(X, adj, edges, dim=args.dim, hidden=args.hidden, epochs=args.epochs)
    Z = model.embed_all(X, adj)
    auc = link_prediction_auc(Z, edges)
    took = time.perf_counter() - t0

    art = _bootstrap.ARTIFACT_DIR
    out = art / "graphsage_embeddings.npz"
    np.savez_compressed(out, claim_ids=np.array(connected), embeddings=Z.astype(np.float32))
    (art / "graphsage_report.json").write_text(
        json.dumps(
            {
                "nodes_trained": len(connected),
                "edges": int(len(edges)),
                "dim": args.dim,
                "epochs": args.epochs,
                "losses": losses,
                "link_prediction_auc": round(auc, 4),
                "seconds": round(took, 1),
            },
            indent=2,
        )
    )
    print(f"\nLink-prediction AUC: {auc:.4f}   ({took:.1f}s)")
    print(f"Saved embeddings -> {out}")
    print("Next: python ml/train_xgboost.py --embeddings artifacts/graphsage_embeddings.npz")


if __name__ == "__main__":
    main()

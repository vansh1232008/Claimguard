# Architecture

## Layers

```
 Next.js dashboard  ──HTTP──►  FastAPI  ──►  InvestigationPipeline (LangGraph)
                                  │                    │
                                  │                    ├─► LLMService      (Gemini | offline reasoner)
                                  │                    ├─► VectorStore     (Qdrant | in-memory)
                                  │                    ├─► GraphStore      (Neo4j  | NetworkX)
                                  │                    ├─► FraudScorer     (XGBoost | scorecard)
                                  │                    ├─► Vision          (YOLOv8 | OpenCV | stats)
                                  │                    └─► Documents       (OCR/pdf | regex)
                                  ▼
                            SQLAlchemy  ──►  PostgreSQL | SQLite
```

Every service in the right-hand column is a façade with a production backend and an offline
fallback behind one interface. The agents never branch on which one is live; only
`/api/system/info` reports it.

## The state machine

`backend/app/agents/pipeline.py` defines the topology once and compiles it two ways: as a
LangGraph `StateGraph` when LangGraph is installed, and as `SimpleRunner` when it is not.
Both walk the same nodes, the same linear edges and the same two conditional edges, so
behaviour does not depend on what happens to be installed.

Two conditional edges:

- **after `intake`** — a filing too incomplete to investigate (completeness < 0.25) skips
  straight to adjudication. There is nothing for the middle agents to work with.
- **after `anomaly`** — a claim with no neighbours in the graph and a probability below 0.30
  skips the network agent. On a typical portfolio that removes a graph traversal and an LLM
  call from most claims.

## State contract

Agents communicate through one dict (`InvestigationState`). Each agent:

- reads what it needs,
- writes its output under its own key,
- appends to three shared accumulators: `risk_signals`, `evidence`, `agent_runs`.

`BaseAgent.__call__` wraps every agent with timing, LLM-call counting and error capture. An
agent that raises is recorded as `status: failed` and the pipeline continues — one broken
agent degrades an investigation, it does not lose it. Only `intake` and `adjudication` are
marked `critical`.

## Feature parity between training and serving

`backend/app/services/features.py` is imported by both the API and the training scripts in
`ml/`. There is exactly one `build_feature_row`. This is deliberate: training/serving skew —
two copies of the feature code drifting apart — is the most common way a fraud model quietly
stops working in production.

`artifacts/feature_metadata.json` stores the feature order used at training time, and the
scorer reads it at load, so adding a feature to the list does not silently misalign an older
model.

## Scoring

```
signal_pressure = Σ w_i / (Σ w_i + 0.75)          # squash into 0..1
fraud_score     = 0.65 · P(fraud | model) + 0.35 · signal_pressure
```

Linear and reconstructible by hand. Two guardrails in `DecisionAgent` sit above the LLM's
output: an uncovered loss is forced to `reject`, and a claim above
`HIGH_VALUE_CLAIM_THRESHOLD` cannot be auto-approved.

## The graph

Nodes are claims. Edges are shared attributes, weighted by how strongly each suggests
collusion:

| Link | Weight | Why |
|---|---|---|
| `SHARED_PHONE`, `SHARED_BANK_ACCOUNT` | 1.0 | unrelated claimants do not share these |
| `SHARED_VEHICLE` | 0.9 | |
| `SHARED_ADDRESS` | 0.8 | households are legitimate, but worth seeing |
| `SHARED_VENDOR` | 0.6 | capped at 25 claims — a busy garage is not a ring |
| `SAME_AREA_SAME_WEEK` | 0.35 | weak on its own, meaningful in combination |

Ring cohesion is the weighted edge density of the connected component, so a component held
together by one weak link scores low even when it is large.

## Persistence

`investigations` and `agent_runs` store the complete audit trail: every agent's output,
duration, LLM usage and risk contribution. An insurer must be able to reconstruct why a
claim was declined months later, and a score in a column cannot do that.

## Extending it

- **A new agent**: subclass `BaseAgent`, set `name` and `state_key`, add it to
  `_build_agents()` and to the edges in `build_langgraph` and `SimpleRunner.linear_next`.
- **A new feature**: add it to `BASE_FEATURES` or `GRAPH_FEATURES`, populate it in
  `build_feature_row`, retrain. Serving picks it up automatically.
- **A new link type**: add it to `LINK_WEIGHTS` and bucket it in `InMemoryGraphStore.rebuild`.
- **A different policy book**: drop markdown into `data/policies/` using
  `## <CLAUSE-ID> <Title>` headings, then `POST /api/system/reindex-policies`.

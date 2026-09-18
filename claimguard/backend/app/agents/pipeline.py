"""The investigation graph.

Topology:

    intake
      │  (abort if the filing is unusable)
      ├──────────────► adjudication
      ▼
    document_analysis ──► damage_analysis ──► coverage ──► anomaly
                                                             │
                              ┌──── network needed? ─────────┤
                              ▼                              ▼
                           network ───────────────────► adjudication

The conditional edge after ``anomaly`` is the point of using a graph rather
than a for-loop: a clean, low-scoring claim with no neighbours in the graph
skips the network agent entirely, which removes an LLM call and a graph
traversal from the majority of claims.

LangGraph drives this when it is installed. When it is not, ``SimpleRunner``
executes exactly the same nodes and edges, so behaviour does not change with
the environment.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from app.agents.anomaly import AnomalyAgent
from app.agents.damage import DamageAgent
from app.agents.decision import DecisionAgent
from app.agents.document_agent import DocumentAgent
from app.agents.intake import IntakeAgent
from app.agents.network import NetworkAgent
from app.agents.policy import PolicyAgent
from app.agents.state import InvestigationState
from app.core.logging import get_logger

logger = get_logger(__name__)

AGENT_ORDER = [
    "intake",
    "document_analysis",
    "damage_analysis",
    "coverage",
    "anomaly",
    "network",
    "adjudication",
]


def _build_agents() -> dict[str, Any]:
    return {
        "intake": IntakeAgent(),
        "document_analysis": DocumentAgent(),
        "damage_analysis": DamageAgent(),
        "coverage": PolicyAgent(),
        "anomaly": AnomalyAgent(),
        "network": NetworkAgent(),
        "adjudication": DecisionAgent(),
    }


# ---------------------------------------------------------------- routing
def route_after_intake(state: InvestigationState) -> str:
    intake = state.get("intake", {}) or {}
    if intake.get("completeness_score") is not None and float(
        intake["completeness_score"]
    ) < 0.25:
        logger.info("Claim %s is too incomplete to investigate; routing to adjudication",
                    state.get("claim_id"))
        return "adjudication"
    return "document_analysis"


def route_after_anomaly(state: InvestigationState) -> str:
    """Only walk the graph when there is something there to walk."""
    graph_ctx = state.get("_graph_context") or {}  # type: ignore[typeddict-item]
    neighbours = graph_ctx.get("neighbours") or []
    prob = float((state.get("anomaly", {}) or {}).get("fraud_probability", 0.0))
    if neighbours or prob >= 0.30:
        return "network"
    return "adjudication"


class SimpleRunner:
    """Dependency-free executor with the same graph semantics as LangGraph."""

    name = "builtin"

    def __init__(self, agents: dict[str, Any]):
        self.agents = agents
        self.routers: dict[str, Callable[[InvestigationState], str]] = {
            "intake": route_after_intake,
            "anomaly": route_after_anomaly,
        }
        self.linear_next = {
            "document_analysis": "damage_analysis",
            "damage_analysis": "coverage",
            "coverage": "anomaly",
            "network": "adjudication",
        }

    def invoke(self, state: InvestigationState) -> InvestigationState:
        node = "intake"
        visited: set[str] = set()
        while node:
            if node in visited:  # cycle guard
                break
            visited.add(node)
            agent = self.agents[node]
            state = agent(state)
            if node == "adjudication":
                break
            router = self.routers.get(node)
            node = router(state) if router else self.linear_next.get(node, "")
        return state


def _as_node(agent: Any) -> Callable[[InvestigationState], InvestigationState]:
    """Wrap an agent instance in a plain function.

    LangGraph inspects a node's call signature to decide how to invoke it, and
    a callable *instance* is not inspected the way a function is - the agent
    ends up receiving something other than the state. A one-line closure keeps
    the node a plain `state -> state` function, which is what LangGraph
    expects.
    """

    def node(state):
        return agent(state)

    # Deliberately unannotated: LangGraph reads a node's type hints to decide
    # the node's input schema, and annotating this with the InvestigationState
    # TypedDict makes it hand the node a filtered state instead of the real one.
    node.__name__ = f"{agent.name}_node"
    return node


def build_langgraph(agents: dict[str, Any]):
    """Compile the same topology as a LangGraph StateGraph."""
    from langgraph.graph import END, StateGraph

    builder = StateGraph(dict)
    for name, agent in agents.items():
        builder.add_node(name, _as_node(agent))

    builder.set_entry_point("intake")
    builder.add_conditional_edges(
        "intake",
        route_after_intake,
        {"document_analysis": "document_analysis", "adjudication": "adjudication"},
    )
    builder.add_edge("document_analysis", "damage_analysis")
    builder.add_edge("damage_analysis", "coverage")
    builder.add_edge("coverage", "anomaly")
    builder.add_conditional_edges(
        "anomaly",
        route_after_anomaly,
        {"network": "network", "adjudication": "adjudication"},
    )
    builder.add_edge("network", "adjudication")
    builder.add_edge("adjudication", END)
    return builder.compile()


class InvestigationPipeline:
    def __init__(self) -> None:
        self.agents = _build_agents()
        try:
            self.runner = build_langgraph(self.agents)
            self.engine = "langgraph"
        except Exception as exc:
            logger.info("LangGraph unavailable (%s); using the built-in runner", exc)
            self.runner = SimpleRunner(self.agents)
            self.engine = "builtin"
        logger.info("Investigation pipeline ready (engine=%s)", self.engine)

    def run(self, state: InvestigationState) -> InvestigationState:
        started = time.perf_counter()
        result = self.runner.invoke(state)
        if not isinstance(result, dict):  # pragma: no cover
            result = state
        result["duration_ms"] = int((time.perf_counter() - started) * 1000)  # type: ignore[typeddict-unknown-key]
        result["engine"] = self.engine  # type: ignore[typeddict-unknown-key]
        return result  # type: ignore[return-value]


_pipeline: InvestigationPipeline | None = None


def get_pipeline() -> InvestigationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InvestigationPipeline()
    return _pipeline

"""
LangGraph Orchestration Graph
==============================
Wires the four agents into a stateful directed graph with conditional
re-summarisation loop.

Flow:
  retrieval → summarization → critique ─┬→ (approved) → writer → END
                                         └→ (needs revision) → summarization (loop)
"""

from __future__ import annotations
import logging

from langgraph.graph import StateGraph, END

from config.schemas import GraphState
from agents import (
    retrieval_agent,
    summarization_agent,
    critique_agent,
    should_re_summarise,
    writer_agent,
)

logger = logging.getLogger(__name__)


def _state_to_dict(state: GraphState) -> dict:
    return state.model_dump()


def _dict_to_state(d: dict) -> GraphState:
    return GraphState(**d)


# LangGraph requires the state type to be a TypedDict or dict.
# We bridge our Pydantic model by wrapping each node.

def _wrap(agent_fn):
    """Wrap a Pydantic-state agent so LangGraph can call it with a plain dict."""
    def wrapped(state_dict: dict) -> dict:
        state = GraphState(**state_dict)
        updated = agent_fn(state)
        return updated.model_dump()
    wrapped.__name__ = agent_fn.__name__
    return wrapped


def _wrap_condition(condition_fn):
    def wrapped(state_dict: dict) -> str:
        state = GraphState(**state_dict)
        return condition_fn(state)
    wrapped.__name__ = condition_fn.__name__
    return wrapped


def build_graph() -> StateGraph:
    """
    Construct and compile the multi-agent LangGraph.

    Returns:
        Compiled LangGraph app ready for .invoke() calls.
    """
    builder = StateGraph(dict)   # state is plain dict; we wrap/unwrap Pydantic

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("retrieval",      _wrap(retrieval_agent))
    builder.add_node("summarization",  _wrap(summarization_agent))
    builder.add_node("critique",       _wrap(critique_agent))
    builder.add_node("writer",         _wrap(writer_agent))

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("retrieval")

    # ── Fixed edges ───────────────────────────────────────────────────────────
    builder.add_edge("retrieval",     "summarization")
    builder.add_edge("summarization", "critique")

    # ── Conditional edge: critique → revise or write ──────────────────────────
    builder.add_conditional_edges(
        "critique",
        _wrap_condition(should_re_summarise),
        {
            "revise": "summarization",   # loop back
            "write":  "writer",
        },
    )

    builder.add_edge("writer", END)

    return builder.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(query: str, paper_ids: list[str] | None = None) -> GraphState:
    """
    Run the full multi-agent research pipeline.

    Args:
        query:     The research question or topic.
        paper_ids: Optional list of paper IDs to restrict retrieval.

    Returns:
        Final GraphState with the completed report.
    """
    app = build_graph()

    initial_state = GraphState(
        query=query,
        paper_ids=paper_ids or [],
        status="running",
    ).model_dump()

    logger.info("=== Pipeline start: '%s' ===", query)
    final_dict = app.invoke(initial_state)
    final_state = GraphState(**final_dict)
    logger.info("=== Pipeline complete: status=%s ===", final_state.status)
    return final_state

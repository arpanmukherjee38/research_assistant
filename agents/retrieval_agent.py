"""
Agent: Retrieval
Queries ChromaDB for relevant chunks given the user's research query.
"""

from __future__ import annotations
import logging

from config.schemas import GraphState
from config.settings import TOP_K_RESULTS
from rag.pipeline import retrieve

logger = logging.getLogger(__name__)


def retrieval_agent(state: GraphState) -> GraphState:
    """
    LangGraph node: retrieve relevant paper chunks from ChromaDB.

    Reads:  state.query, state.paper_ids
    Writes: state.retrieval_result, state.status
    """
    logger.info("[Retrieval Agent] Query: %s", state.query)

    try:
        result = retrieve(
            query=state.query,
            top_k=TOP_K_RESULTS,
            paper_ids=state.paper_ids or None,
        )
        state.retrieval_result = result
        state.status = "retrieved"
        logger.info("[Retrieval Agent] Found %d chunks", result.total_found)
    except Exception as exc:
        state.error = f"Retrieval failed: {exc}"
        state.status = "error"
        logger.error("[Retrieval Agent] Error: %s", exc)

    return state

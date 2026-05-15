"""
Shared Pydantic schemas used across all agents and the graph state.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ── Paper metadata ─────────────────────────────────────────────────────────────

class PaperMetadata(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    abstract: str = ""
    year: str = ""
    source: str = "unknown"   # "arxiv" | "pdf_upload"
    url: str = ""


# ── Retrieval ──────────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    text: str
    score: float            # cosine similarity (0-1)
    metadata: dict[str, Any] = {}


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    total_found: int


# ── Summarisation ─────────────────────────────────────────────────────────────

class PaperSummary(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    year: str = ""
    url: str = ""
    key_contributions: list[str] = Field(default_factory=list)
    methodology: str = ""
    results: str = ""
    limitations: str = ""
    relevance_to_query: str = ""
    full_summary: str = ""


# ── Critique ──────────────────────────────────────────────────────────────────

class CritiqueResult(BaseModel):
    paper_id: str
    quality_score: int          # 1-10
    completeness_score: int     # 1-10
    accuracy_flags: list[str] = []   # potential inaccuracies flagged
    suggestions: list[str] = []
    approved: bool              # True if quality_score >= threshold


# ── Final report ──────────────────────────────────────────────────────────────

class ResearchReport(BaseModel):
    query: str
    total_papers: int
    summaries: list[PaperSummary]
    critiques: list[CritiqueResult]
    synthesis: str              # the writer agent's combined narrative
    key_themes: list[str] = []
    research_gaps: list[str] = []
    recommended_reading: list[str] = []
    markdown_path: str = ""


# ── LangGraph shared state ────────────────────────────────────────────────────

class GraphState(BaseModel):
    """Mutable state passed between all nodes in the LangGraph."""
    query: str = ""
    paper_ids: list[str] = []
    papers: list[PaperMetadata] = []
    retrieval_result: RetrievalResult | None = None
    summaries: list[PaperSummary] = []
    critiques: list[CritiqueResult] = []
    critique_round: int = 0
    report: ResearchReport | None = None
    error: str = ""
    status: str = "idle"        # idle | running | done | error

    class Config:
        arbitrary_types_allowed = True

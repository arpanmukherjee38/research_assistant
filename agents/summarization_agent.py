from __future__ import annotations
import json
import logging
import os

from langchain_ollama import ChatOllama
from config.schemas import GraphState, PaperSummary, RetrievedChunk
from config.settings import OLLAMA_MODEL_PRO

logger = logging.getLogger(__name__)

# Enforce JSON formatting natively in Ollama
_model = ChatOllama(
    model=OLLAMA_MODEL_PRO, 
    temperature=0.3, 
    format="json",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

SUMMARY_PROMPT = """\
You are an expert academic research assistant. Given the following excerpts from a research paper, produce a structured summary.
Paper title: {title}
Authors: {authors}
Year: {year}
Research query: {query}

Paper excerpts:
{excerpts}

Return your response as valid JSON matching this schema exactly:
{{
  "key_contributions": ["...", "..."], "methodology": "...",
  "results": "...", "limitations": "...",
  "relevance_to_query": "...", "full_summary": "..."
}}
"""

def _group_chunks_by_paper(chunks: list[RetrievedChunk]) -> dict[str, list[RetrievedChunk]]:
    groups: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks: groups.setdefault(chunk.paper_id, []).append(chunk)
    return groups

def _summarise_paper(paper_id: str, chunks: list[RetrievedChunk], query: str, paper_meta: dict) -> PaperSummary:
    excerpts = "\n\n---\n\n".join(c.text for c in chunks[:6]) 
    title   = paper_meta.get("title", chunks[0].title)
    authors = paper_meta.get("authors", chunks[0].metadata.get("authors", ""))
    year    = paper_meta.get("year", chunks[0].metadata.get("year", ""))

    prompt = SUMMARY_PROMPT.format(title=title, authors=authors, year=year, query=query, excerpts=excerpts)

    try:
        response = _model.invoke(prompt)
        data = json.loads(response.content.strip())
    except Exception as exc:
        logger.warning("[Summarization] JSON parse failed for %s: %s", paper_id, exc)
        data = {
            "key_contributions": [], "methodology": "", "results": "",
            "limitations": "", "relevance_to_query": "",
            "full_summary": f"Error generating summary: {exc}",
        }

    return PaperSummary(
        paper_id=paper_id, title=title, year=str(year), url=paper_meta.get("url", ""),
        authors=authors.split(", ") if isinstance(authors, str) else authors,
        **data,
    )

def summarization_agent(state: GraphState) -> GraphState:
    if not state.retrieval_result or not state.retrieval_result.chunks:
        state.error = "No retrieval results to summarise."
        state.status = "error"
        return state

    paper_lookup = {p.paper_id: {"title": p.title, "authors": p.authors, "year": p.year, "url": p.url} for p in (state.papers or [])}
    grouped = _group_chunks_by_paper(state.retrieval_result.chunks)
    
    state.summaries = [_summarise_paper(pid, chunks, state.query, paper_lookup.get(pid, {})) for pid, chunks in grouped.items()]
    state.status = "summarised"
    return state
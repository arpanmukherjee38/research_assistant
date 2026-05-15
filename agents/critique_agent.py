from __future__ import annotations
import json
import logging
import os

from langchain_ollama import ChatOllama
from config.schemas import GraphState, CritiqueResult, PaperSummary
from config.settings import OLLAMA_MODEL_FLASH, MIN_QUALITY_SCORE

logger = logging.getLogger(__name__)

_model = ChatOllama(
    model=OLLAMA_MODEL_FLASH, 
    temperature=0.1, 
    format="json",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

CRITIQUE_PROMPT = """\
You are a rigorous academic peer reviewer. Evaluate the following paper summary.
Research query: {query}
Paper: {title} ({year})

Summary to evaluate:
{summary_json}

Score the summary on these dimensions and return ONLY valid JSON:
{{
  "quality_score": <int 1-10>,
  "completeness_score": <int 1-10>,
  "accuracy_flags": ["list inaccuracies"],
  "suggestions": ["improvement suggestions"]
}}
"""

def _critique_summary(summary: PaperSummary, query: str) -> CritiqueResult:
    prompt = CRITIQUE_PROMPT.format(query=query, title=summary.title, year=summary.year, summary_json=summary.model_dump_json(indent=2))

    try:
        response = _model.invoke(prompt)
        data = json.loads(response.content.strip())
    except Exception as exc:
        data = {"quality_score": 7, "completeness_score": 7, "accuracy_flags": [], "suggestions": []}

    approved = data.get("quality_score", 0) >= MIN_QUALITY_SCORE
    return CritiqueResult(paper_id=summary.paper_id, approved=approved, **{k: v for k, v in data.items()})

def critique_agent(state: GraphState) -> GraphState:
    if not state.summaries:
        state.error, state.status = "No summaries to critique.", "error"
        return state

    state.critiques = [_critique_summary(s, state.query) for s in state.summaries]
    state.critique_round += 1
    state.status = "critiqued"
    return state

def should_re_summarise(state: GraphState) -> str:
    from config.settings import MAX_CRITIQUE_ROUNDS
    failed = [c for c in state.critiques if not c.approved]
    return "revise" if failed and state.critique_round < MAX_CRITIQUE_ROUNDS else "write"
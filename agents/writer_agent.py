from __future__ import annotations
import logging
import os
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama
from config.schemas import GraphState, ResearchReport
from config.settings import OLLAMA_MODEL_PRO, OUTPUT_DIR

logger = logging.getLogger(__name__)

_model = ChatOllama(
    model=OLLAMA_MODEL_PRO, 
    temperature=0.4,
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

WRITER_PROMPT = """\
You are a senior research analyst writing a comprehensive literature review.
Research query: {query}
Synthesise the following summaries into a well-structured report.

Paper summaries:
{summaries_block}

Write the following sections (use Markdown headings):
## Executive summary
## Key themes
## Paper-by-paper highlights
## Research gaps
## Recommended reading order
## Conclusion
"""

def writer_agent(state: GraphState) -> GraphState:
    if not state.summaries:
        state.error, state.status = "No summaries available.", "error"
        return state

    lines = [f"### {s.title} ({s.year})\nAuthors: {', '.join(s.authors)}\nFull summary: {s.full_summary}\nKey contributions: {'; '.join(s.key_contributions)}\nMethodology: {s.methodology}\nResults: {s.results}\nLimitations: {s.limitations}\n" for s in state.summaries]
    
    prompt = WRITER_PROMPT.format(query=state.query, summaries_block="\n".join(lines))
    synthesis = _model.invoke(prompt).content.strip()

    citations = [f"- [{s.title}]({s.url}) — {', '.join(s.authors[:2])} ({s.year})" if s.url else f"- {s.title} — {', '.join(s.authors[:2])} ({s.year})" for s in state.summaries]
    
    full_report_md = f"# Research Report: {state.query}\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n*Papers analysed: {len(state.summaries)}*\n---\n\n{synthesis}\n\n---\n## References\n" + "\n".join(citations)

    key_themes = [line.strip()[2:] for line in synthesis.split("\n") if line.strip().startswith("- ") and len(line.strip()) < 120][:6]

    md_path = OUTPUT_DIR / f"report_{state.query[:20].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    md_path.write_text(full_report_md, encoding="utf-8")

    state.report = ResearchReport(
        query=state.query, total_papers=len(state.summaries), summaries=state.summaries,
        critiques=state.critiques, synthesis=synthesis, key_themes=key_themes, markdown_path=str(md_path),
    )
    state.status = "done"
    return state
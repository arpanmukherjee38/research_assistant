from __future__ import annotations
import logging
from pathlib import Path
import arxiv

from config.schemas import PaperMetadata
from config.settings import ARXIV_MAX_RESULTS, ARXIV_SORT_BY, OUTPUT_DIR

logger = logging.getLogger(__name__)

PDF_DIR = OUTPUT_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

def search_arxiv(query: str, max_results: int = ARXIV_MAX_RESULTS) -> list[PaperMetadata]:
    sort_criterion = arxiv.SortCriterion.Relevance
    if ARXIV_SORT_BY == "lastUpdatedDate":
        sort_criterion = arxiv.SortCriterion.LastUpdatedDate
    elif ARXIV_SORT_BY == "submittedDate":
        sort_criterion = arxiv.SortCriterion.SubmittedDate

    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criterion
    )

    papers: list[PaperMetadata] = []
    try:
        for result in client.results(search):
            papers.append(PaperMetadata(
                paper_id=result.get_short_id(),
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary.replace("\n", " "),
                year=str(result.published.year),
                source="arxiv",
                url=result.entry_id
            ))
    except Exception as exc:
        logger.error("arXiv search failed: %s", exc)
    return papers

def download_arxiv_pdf(paper_id: str) -> Path | None:
    client = arxiv.Client(delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(id_list=[paper_id])
    
    try:
        result = next(client.results(search))
        safe_id = paper_id.replace("/", "_") 
        filename = f"{safe_id}.pdf"
        filepath = PDF_DIR / filename
        
        if not filepath.exists():
            result.download_pdf(dirpath=str(PDF_DIR), filename=filename)
        return filepath
    except Exception as exc:
        logger.error("Failed to download PDF for %s: %s", paper_id, exc)
        return None
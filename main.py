"""
CLI entrypoint — run the full pipeline from the terminal.

Usage:
    python main.py --query "transformer attention mechanisms" --arxiv --n 3
    python main.py --query "RAG vs fine-tuning" --pdf path/to/paper.pdf
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import track

console = Console()
logging.basicConfig(level=logging.WARNING)  # suppress verbose logs in CLI

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant CLI")
    parser.add_argument("--query",  required=True, help="Research question")
    parser.add_argument("--arxiv",  action="store_true", help="Search and ingest from arXiv")
    parser.add_argument("--n",      type=int, default=3, help="Number of arXiv papers to fetch")
    parser.add_argument("--pdf",    type=str, default=None, help="Path to a local PDF to ingest")
    parser.add_argument("--title",  type=str, default="Uploaded Paper", help="Title for PDF")
    parser.add_argument("--no-pdf-download", action="store_true",
                        help="Use abstracts only (faster, no PDF download)")
    args = parser.parse_args()

    from config.settings import GEMINI_API_KEY
    if not GEMINI_API_KEY:
        console.print("[red]❌ GEMINI_API_KEY not set in .env[/red]")
        sys.exit(1)

    from tools.arxiv_tool import search_arxiv, download_arxiv_pdf
    from tools.pdf_tool import extract_text_from_pdf
    from rag.pipeline import ingest_paper, ingest_abstract_only, collection_count
    from graph.orchestrator import run_pipeline
    from config.schemas import PaperMetadata

    console.rule("[bold]Multi-Agent Research Assistant[/bold]")
    console.print(f"[cyan]Query:[/cyan] {args.query}\n")

    # ── Ingest ────────────────────────────────────────────────────────────────
    if args.arxiv:
        console.print(f"[yellow]Searching arXiv for:[/yellow] {args.query}")
        papers = search_arxiv(args.query, max_results=args.n)
        console.print(f"Found [green]{len(papers)}[/green] papers\n")

        for paper in track(papers, description="Ingesting papers…"):
            if not args.no_pdf_download:
                pdf_path = download_arxiv_pdf(paper.paper_id)
                if pdf_path:
                    text = extract_text_from_pdf(pdf_path)
                    ingest_paper(paper, text)
                    continue
            ingest_abstract_only(paper)

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            console.print(f"[red]PDF not found: {pdf_path}[/red]")
            sys.exit(1)
        paper = PaperMetadata(
            paper_id=pdf_path.stem,
            title=args.title,
            source="pdf_upload",
        )
        text = extract_text_from_pdf(pdf_path)
        ingest_paper(paper, text)
        console.print(f"[green]Ingested:[/green] {pdf_path.name}")

    count = collection_count()
    if count == 0:
        console.print("[red]No chunks in vector store. Use --arxiv or --pdf to add papers first.[/red]")
        sys.exit(1)

    console.print(f"\n[dim]Vector store: {count} chunks[/dim]")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    console.print("\n[bold]Running multi-agent pipeline…[/bold]")
    with console.status("Agents working…"):
        state = run_pipeline(query=args.query)

    if state.status == "error":
        console.print(f"[red]Pipeline error: {state.error}[/red]")
        sys.exit(1)

    report = state.report

    # ── Print results ─────────────────────────────────────────────────────────
    console.print(Panel(
        f"Papers: {report.total_papers}  |  "
        f"Critique rounds: {state.critique_round}  |  "
        f"Report saved: {Path(report.markdown_path).name}",
        title="[green]Pipeline complete[/green]",
    ))

    if report.key_themes:
        console.print("\n[bold]Key themes:[/bold]")
        for t in report.key_themes:
            console.print(f"  • {t}")

    console.print("\n[bold]Synthesis:[/bold]")
    console.print(Markdown(report.synthesis[:2000] + ("…" if len(report.synthesis) > 2000 else "")))

    console.print(f"\n[dim]Full report saved to:[/dim] {report.markdown_path}")


if __name__ == "__main__":
    main()

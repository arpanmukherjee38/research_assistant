"""
Streamlit UI for the Multi-Agent AI Research Assistant.
Run from the project root with: streamlit run ui/app.py
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path

# ── Robust path fix (works on Windows, macOS, Linux) ─────────────────────────
# __file__ = .../research_assistant/ui/app.py
# ROOT     = .../research_assistant
ROOT = Path(__file__).resolve().parent.parent

# Insert at position 0 so our packages shadow any installed packages with the same name
for _p in [str(ROOT), str(ROOT / "config"), str(ROOT / "tools"),
           str(ROOT / "rag"), str(ROOT / "agents"), str(ROOT / "graph")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Also set working directory to ROOT so relative imports work the same
import os
os.chdir(ROOT)

import streamlit as st

from config.schemas import GraphState
from tools.arxiv_tool import search_arxiv, download_arxiv_pdf
from tools.pdf_tool import extract_text_from_bytes, extract_text_from_pdf
from rag.pipeline import ingest_paper, ingest_abstract_only, collection_count
from graph.orchestrator import run_pipeline
from config.schemas import PaperMetadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("Multi-Agent · RAG · ChromaDB · Gemini")
    st.divider()

    st.subheader("Vector store")
    count = collection_count()
    st.metric("Chunks indexed", count)

    st.divider()
    st.subheader("Add papers")

    ingest_mode = st.radio("Source", ["arXiv search", "Upload PDF"], horizontal=True)

    if ingest_mode == "arXiv search":
        arxiv_query = st.text_input("Search arXiv", placeholder="e.g. retrieval augmented generation")
        n_papers    = st.slider("Max papers", 1, 10, 3)
        full_pdf    = st.checkbox("Download full PDFs", value=False,
                                  help="Slower but gives richer context. Uncheck to use abstracts only.")

        if st.button("🔍 Fetch & Ingest", use_container_width=True):
            if not arxiv_query:
                st.warning("Enter a search query.")
            else:
                with st.spinner("Searching arXiv…"):
                    papers = search_arxiv(arxiv_query, max_results=n_papers)
                st.success(f"Found {len(papers)} papers")

                progress = st.progress(0)
                for i, paper in enumerate(papers):
                    with st.spinner(f"Ingesting: {paper.title[:50]}…"):
                        if full_pdf:
                            pdf_path = download_arxiv_pdf(paper.paper_id)
                            if pdf_path:
                                text = extract_text_from_pdf(pdf_path)
                                ingest_paper(paper, text)
                            else:
                                ingest_abstract_only(paper)
                        else:
                            ingest_abstract_only(paper)
                    progress.progress((i + 1) / len(papers))

                st.success(f"✅ {len(papers)} papers ingested")
                st.rerun()

    else:  # Upload PDF
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        paper_title  = st.text_input("Paper title")
        paper_author = st.text_input("Authors (comma-separated)")
        paper_year   = st.text_input("Year", value="2024")

        if st.button("⬆️ Upload & Ingest", use_container_width=True):
            if not uploaded:
                st.warning("Select a PDF file.")
            elif not paper_title:
                st.warning("Enter the paper title.")
            else:
                with st.spinner("Extracting and ingesting…"):
                    text = extract_text_from_bytes(uploaded.read(), uploaded.name)
                    paper = PaperMetadata(
                        paper_id=uploaded.name.replace(".pdf", ""),
                        title=paper_title,
                        authors=[a.strip() for a in paper_author.split(",")],
                        year=paper_year,
                        source="pdf_upload",
                    )
                    ingest_paper(paper, text)
                st.success("✅ Paper ingested")
                st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Multi-Agent AI Research Assistant")
st.caption("Ask a research question — the agent pipeline retrieves, summarises, critiques, and writes a report.")

query = st.text_area(
    "Research question",
    placeholder="e.g. How do retrieval-augmented generation systems compare to fine-tuning for domain adaptation?",
    height=90,
)

col1, col2 = st.columns([3, 1])
with col2:
    run_btn = st.button("▶ Run pipeline", type="primary", use_container_width=True)

if run_btn:
    if not query.strip():
        st.warning("Enter a research question.")
    elif collection_count() == 0:
        st.warning("No papers indexed yet. Add papers using the sidebar first.")
    else:
        with st.status("Running multi-agent pipeline…", expanded=True) as status_box:
            st.write("🔎 Retrieval agent querying ChromaDB…")
            try:
                final: GraphState = run_pipeline(query=query.strip())
                st.write("📝 Summarization agent processing papers…")
                st.write("🧐 Critique agent evaluating quality…")
                st.write("✍️ Writer agent composing report…")
                status_box.update(label="Pipeline complete!", state="complete")
            except Exception as exc:
                status_box.update(label=f"Pipeline error: {exc}", state="error")
                st.error(str(exc))
                st.stop()

        if final.status == "error":
            st.error(f"Pipeline error: {final.error}")
        else:
            report = final.report

            # ── Report header ──────────────────────────────────────────────
            st.divider()
            st.subheader(f"📄 Report: {query[:80]}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Papers analysed",   report.total_papers)
            m2.metric("Critique rounds",   final.critique_round)
            approved = sum(1 for c in report.critiques if c.approved)
            m3.metric("Summaries approved", f"{approved}/{report.total_papers}")

            # ── Key themes ─────────────────────────────────────────────────
            if report.key_themes:
                st.subheader("🏷️ Key themes")
                theme_cols = st.columns(min(len(report.key_themes), 3))
                for i, theme in enumerate(report.key_themes):
                    theme_cols[i % 3].info(theme)

            # ── Synthesis ──────────────────────────────────────────────────
            st.subheader("📑 Full synthesis")
            st.markdown(report.synthesis)

            # ── Per-paper details ──────────────────────────────────────────
            st.subheader("🔬 Paper summaries")
            for summary in report.summaries:
                critique = next(
                    (c for c in report.critiques if c.paper_id == summary.paper_id), None
                )
                quality = critique.quality_score if critique else "—"
                badge = "✅" if (critique and critique.approved) else "⚠️"

                with st.expander(f"{badge} {summary.title} ({summary.year})  ·  Quality: {quality}/10"):
                    if summary.url:
                        st.markdown(f"[arXiv link]({summary.url})")
                    st.markdown(f"**Authors:** {', '.join(summary.authors)}")

                    tcols = st.columns(2)
                    with tcols[0]:
                        st.markdown("**Key contributions**")
                        for contrib in summary.key_contributions:
                            st.markdown(f"- {contrib}")
                        st.markdown(f"**Methodology:** {summary.methodology}")

                    with tcols[1]:
                        st.markdown(f"**Results:** {summary.results}")
                        st.markdown(f"**Limitations:** {summary.limitations}")
                        st.markdown(f"**Relevance:** {summary.relevance_to_query}")

                    if critique:
                        st.caption(
                            f"Critique — completeness: {critique.completeness_score}/10  |  "
                            f"Flags: {len(critique.accuracy_flags)}  |  "
                            f"Suggestions: {len(critique.suggestions)}"
                        )

            # ── Download ───────────────────────────────────────────────────
            if report.markdown_path and Path(report.markdown_path).exists():
                st.divider()
                md_content = Path(report.markdown_path).read_text(encoding="utf-8")
                st.download_button(
                    label="⬇️ Download Markdown report",
                    data=md_content,
                    file_name=Path(report.markdown_path).name,
                    mime="text/markdown",
                )
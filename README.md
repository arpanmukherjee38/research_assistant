# Multi-Agent AI Research Assistant

> Automated academic paper summarisation powered by **Local Ollama Models**, **LangGraph**, **RAG**, and **ChromaDB** — running entirely offline in a secure Docker container. Your research data never leaves your machine.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-blueviolet?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Privacy](https://img.shields.io/badge/100%25-Local%20%26%20Private-brightgreen?style=flat-square)

---

## Overview

This project implements a **multi-agent research pipeline** that can:

- Search and download papers from **arXiv** or accept PDF uploads
- Chunk and embed documents into a local **ChromaDB** vector store
- Answer research questions using a **4-agent LangGraph pipeline**: Retrieval → Summarization → Critique → Writer
- Self-critique and revise summaries until they meet a configurable quality threshold
- Output a polished **Markdown literature review** to your local `output/` folder

Because everything runs through **Ollama** with locally-hosted open-source models, there is **no API key required** and no data ever sent to an external server.

---

## 🎥 Demo

https://github.com/arpanmukherjee38/research_assistant/videos/demo.mp4

## Architecture

```
User query
    │
    ▼
LangGraph Orchestrator
    │
    ├─→ Retrieval Agent      — queries local ChromaDB for relevant chunks
    ├─→ Summarization Agent  — Local LLM produces structured summaries
    ├─→ Critique Agent       — Fast local LLM scores quality, loops if below threshold
    └─→ Writer Agent         — Synthesises final Markdown report
             │
             ▼
        output/report_*.md
```

### Agent Details

| Agent | Role | Model |
|---|---|---|
| **Retrieval** | Converts query → embedding, cosine similarity search in ChromaDB, returns top-k chunks | `nomic-embed-text` |
| **Summarization** | Groups chunks by paper, produces structured JSON (contributions, methodology, results, limitations, relevance) | `qwen2.5:3b` |
| **Critique** | Scores each summary 1–10, rejects below `MIN_QUALITY_SCORE`, sends back for revision (up to `MAX_CRITIQUE_ROUNDS`) | `qwen2.5:3b` |
| **Writer** | Synthesises all approved summaries into a literature review with executive summary, key themes, research gaps, and recommended reading order | `qwen2.5:3b` |

---

## Tech Stack

| Layer | Software |
|---|---|
| LLM | Local Ollama (`qwen2.5:3b`, `llama3.1`) |
| Orchestration | LangGraph |
| RAG framework | LangChain |
| Vector store | ChromaDB (local persistent volume) |
| Embeddings | Ollama `nomic-embed-text` |
| PDF parsing | PyMuPDF (`fitz`) |
| Paper discovery | arXiv API (`arxiv` library) |
| UI | Streamlit |
| Deployment | Docker & Docker Compose |

---

## Project Structure

```
research_assistant/
├── config/
│   ├── settings.py       # all env-driven config (Ollama models, chunks)
│   └── schemas.py        # Pydantic models (GraphState, PaperSummary, …)
├── tools/
│   ├── arxiv_tool.py     # arXiv search + PDF download
│   └── pdf_tool.py       # PyMuPDF text extraction
├── rag/
│   └── pipeline.py       # chunk → local embed → store → retrieve
├── agents/
│   ├── retrieval_agent.py
│   ├── summarization_agent.py
│   ├── critique_agent.py
│   └── writer_agent.py
├── graph/
│   └── orchestrator.py   # LangGraph graph definition
├── ui/
│   └── app.py            # Streamlit web UI
├── output/               # generated reports + PDFs (Docker Volume Sync)
├── videos/               # project demonstration videos
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Quickstart (Docker)

No API keys needed — everything runs locally via Ollama.

### 1. Prerequisites

- Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- *(Optional but recommended)* If you have an NVIDIA GPU, configure Docker to use it for significantly faster generation

### 2. Launch the app

```bash
docker-compose up --build
```

> **Note on first run:** The Auto-Puller container will automatically download the required AI models (`nomic-embed-text`, `llama3.1`, `qwen2.5:3b`) to a persistent Docker volume. This is a multi-gigabyte download and may take some time. Subsequent launches start in seconds.

### 3. Open the UI

Once the terminal logs show `All models ready!`, open your browser and navigate to:

```
http://localhost:8505
```

Use the sidebar to add papers (arXiv search or PDF upload), enter a research question, and click **Run pipeline**.

---

## Configuration

Edit `config/settings.py` to customise models and pipeline behaviour:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL_PRO` | `qwen2.5:3b` | Main reasoning model for Summarization and Writing |
| `OLLAMA_MODEL_FLASH` | `qwen2.5:3b` | Fast model for the Critique loop |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Local embedding model |
| `CHUNK_SIZE` | `1000` | Characters per RAG chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `TOP_K_RESULTS` | `6` | Chunks retrieved per query |
| `MIN_QUALITY_SCORE` | `7` | Minimum critique score before re-summarisation |
| `MAX_CRITIQUE_ROUNDS` | `2` | Maximum revision loops per summary |
| `ARXIV_MAX_RESULTS` | `5` | Default papers fetched from arXiv |

To use a larger model (e.g. `llama3.1:8b`) for higher-quality summaries, update `OLLAMA_MODEL_PRO` and ensure it is pulled via Ollama before running the pipeline.

---

## Privacy & Security

- All LLM inference runs **locally** via Ollama — no data is sent to OpenAI, Anthropic, or any third-party API
- ChromaDB persists to a **local Docker volume** — your indexed papers stay on your machine
- Generated reports are written directly to your `output/` folder via Docker volume mounts

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
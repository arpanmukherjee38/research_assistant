from __future__ import annotations
import hashlib
import logging
import os
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from config.settings import OLLAMA_EMBEDDING_MODEL, CHROMA_DIR, CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS
from config.schemas import PaperMetadata, RetrievedChunk, RetrievalResult

logger = logging.getLogger(__name__)

# Connect to the local Docker Ollama instance
_embedder = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL,
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " ", ""])

def ingest_paper(paper: PaperMetadata, full_text: str) -> int:
    if not full_text.strip(): return 0

    collection = _get_collection()
    chunks = _splitter.split_text(full_text)
    ids, documents, metadatas = [], [], []

    for i, chunk in enumerate(chunks):
        ids.append(hashlib.md5(f"{paper.paper_id}_{i}".encode()).hexdigest())
        documents.append(chunk)
        metadatas.append({
            "paper_id": paper.paper_id, "title": paper.title,
            "authors": ", ".join(paper.authors[:3]), "year": paper.year,
            "source": paper.source, "url": paper.url, "chunk_idx": i,
        })

    # Batch process to respect local RAM limits
    batch_size = 50 
    all_embeddings: list[list[float]] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        all_embeddings.extend(_embedder.embed_documents(batch))

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=all_embeddings)
    return len(chunks)

def ingest_abstract_only(paper: PaperMetadata) -> int:
    return ingest_paper(paper, paper.abstract)

def retrieve(query: str, top_k: int = TOP_K_RESULTS, paper_ids: list[str] | None = None) -> RetrievalResult:
    collection = _get_collection()
    query_embedding = _embedder.embed_query(query)
    where = {"paper_id": {"$in": paper_ids}} if paper_ids else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        score = 1.0 - results["distances"][0][i]
        chunks.append(RetrievedChunk(
            chunk_id=results["ids"][0][i], paper_id=meta.get("paper_id", ""),
            title=meta.get("title", ""), text=doc, score=round(score, 4), metadata=meta,
        ))

    chunks.sort(key=lambda c: c.score, reverse=True)
    return RetrievalResult(query=query, chunks=chunks, total_found=len(chunks))

def collection_count() -> int:
    return _get_collection().count()
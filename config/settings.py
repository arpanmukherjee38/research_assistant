import os
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = BASE_DIR / "output"
CHROMA_DIR  = BASE_DIR / ".chromadb"

OUTPUT_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# --- OLLAMA LOCAL MODELS ---
OLLAMA_MODEL_PRO       = "qwen2.5:3b"             
OLLAMA_MODEL_FLASH     = "qwen2.5:3b"           
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"   

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 6

CHROMA_COLLECTION = "research_papers_local"

MAX_SUMMARY_TOKENS   = 1024
MAX_CRITIQUE_ROUNDS  = 2
MIN_QUALITY_SCORE    = 7

ARXIV_MAX_RESULTS    = 5
ARXIV_SORT_BY        = "relevance"
"""Configuration for the AscendMind knowledge base."""

import os

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Source Markdown (produced by MinerU from the official CANN PDF)
MD_PATH = os.environ.get(
    "ASCEND_KB_MD_PATH",
    os.path.join(PROJECT_DIR, "data", "ascend_c_guide.md"),
)

# Index storage
INDEX_DIR = os.path.join(BASE_DIR, "index_store")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

# ── Sparse index (TF-IDF) ─────────────────────────────────────────────
TFIDF_PATH = os.path.join(INDEX_DIR, "tfidf.pkl")

# ── Dense index (FAISS, optional) ──────────────────────────────────────
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_MAX_CHARS = int(os.environ.get("ASCEND_KB_CHUNK_MAX", "1500"))
CHUNK_MIN_CHARS = int(os.environ.get("ASCEND_KB_CHUNK_MIN", "100"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("ASCEND_KB_CHUNK_OVERLAP", "200"))

# ── Embedding (for dense mode) ─────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get(
    "ASCEND_KB_EMBED_MODEL",
    "BAAI/bge-base-zh-v1.5",
)
EMBEDDING_DIM = 768
BATCH_SIZE = int(os.environ.get("ASCEND_KB_BATCH_SIZE", "64"))

# ── Retrieval ──────────────────────────────────────────────────────────
DEFAULT_TOP_K = int(os.environ.get("ASCEND_KB_TOP_K", "5"))

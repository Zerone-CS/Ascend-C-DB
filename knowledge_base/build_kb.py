#!/usr/bin/env python3
"""Build the Ascend C knowledge base: parse Markdown -> chunk -> index.

Default mode: TF-IDF sparse retrieval (fast, no GPU needed).
Optional: --dense to also build FAISS dense index with BGE embeddings.

Usage:
    python -m knowledge_base.build_kb            # sparse only
    python -m knowledge_base.build_kb --dense    # sparse + dense
    python -m knowledge_base.build_kb --md /path/to/custom.md
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer_utils import jieba_tokenizer

import numpy as np

from kb_config import (
    BATCH_SIZE,
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNKS_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    MD_PATH,
    META_PATH,
    TFIDF_PATH,
)

log = logging.getLogger(__name__)


SKIP_TITLES = {
    "\u5546\u6807\u58f0\u660e", "\u6ce8\u610f", "\u5b89\u5168\u58f0\u660e",
    "\u4ea7\u54c1\u751f\u547d\u5468\u671f\u653f\u7b56", "\u6f0f\u6d1e\u5904\u7406\u6d41\u7a0b",
    "\u534e\u4e3a\u521d\u59cb\u8bc1\u4e66\u6743\u8d23\u8bf4\u660e",
    "\u534e\u4e3a\u4f01\u4e1a\u4e1a\u52a1\u6700\u7ec8\u7528\u6237\u8bb8\u53ef\u534f\u8bae(EULA)",
    "\u4ea7\u54c1\u8d44\u6599\u751f\u547d\u5468\u671f\u7b56\u7565",
    "\u76ee \u5f55",
    "CANN \u5546\u7528\u7248 8.2.RC1",
}


def read_markdown(path: str) -> str:
    """Read a Markdown file and return its content."""
    if not os.path.exists(path):
        log.error("Markdown \u6587\u4ef6\u4e0d\u5b58\u5728: %s", path)
        log.error("\u8bf7\u786e\u4fdd\u5df2\u8fd0\u884c MinerU \u89e3\u6790 PDF\uff0c\u6216\u901a\u8fc7 --md \u6307\u5b9a\u81ea\u5b9a\u4e49\u8def\u5f84")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def split_by_headers(text: str) -> list[dict]:
    """Split Markdown text by top-level headers."""
    lines = text.split("\n")
    sections = []
    current_title = ""
    current_lines = []
    current_line_no = 1

    for idx, line in enumerate(lines, start=1):
        if line.startswith("# "):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({
                        "title": current_title,
                        "body": body,
                        "line": current_line_no,
                    })
            current_title = line.lstrip("# ").strip()
            current_lines = []
            current_line_no = idx
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({
                "title": current_title,
                "body": body,
                "line": current_line_no,
            })
    return sections


def merge_small_sections(sections: list[dict], min_chars: int) -> list[dict]:
    """Merge sections smaller than *min_chars* into their predecessor."""
    if not sections:
        return sections
    merged = [sections[0]]
    for sec in sections[1:]:
        if len(sec["body"]) < min_chars and merged:
            merged[-1]["body"] += "\n\n" + sec["title"] + "\n" + sec["body"]
            if not merged[-1]["title"]:
                merged[-1]["title"] = sec["title"]
        else:
            merged.append(sec)
    return merged


def split_long_section(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split a long section into overlapping chunks by paragraph boundary."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = tail + "\n\n" + para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_chunks(sections: list[dict]) -> list[dict]:
    """Convert sections into indexed chunks ready for retrieval."""
    chunks = []
    for sec in sections:
        title = sec["title"]
        if title in SKIP_TITLES:
            continue
        full_text = (title + "\n\n" + sec["body"]) if title else sec["body"]
        if len(full_text) <= CHUNK_MAX_CHARS:
            chunks.append({
                "id": len(chunks),
                "title": title,
                "text": full_text,
                "line": sec["line"],
            })
        else:
            sub_texts = split_long_section(full_text, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
            for sub_idx, sub in enumerate(sub_texts):
                chunks.append({
                    "id": len(chunks),
                    "title": f"{title} (part {sub_idx + 1})",
                    "text": sub,
                    "line": sec["line"],
                })
    return chunks


def build_tfidf_index(chunks: list[dict]) -> None:
    """Build and persist a TF-IDF sparse index."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    log.info("\u6784\u5efa TF-IDF \u7d22\u5f15 (jieba \u5206\u8bcd) ...")
    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(
        tokenizer=jieba_tokenizer,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    with open(TFIDF_PATH, "wb") as fh:
        pickle.dump({"vectorizer": vectorizer, "matrix": tfidf_matrix}, fh)
    log.info("TF-IDF \u7d22\u5f15\u5df2\u4fdd\u5b58: %s", TFIDF_PATH)
    log.info("\u8bcd\u8868\u5927\u5c0f: %d, \u77e9\u9635\u5f62\u72b6: %s", len(vectorizer.vocabulary_), tfidf_matrix.shape)


def build_dense_index(chunks: list[dict]) -> None:
    """Build and persist a FAISS dense index using BGE embeddings."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("\u7f3a\u5c11\u4f9d\u8d56: pip install -r requirements-dense.txt")
        sys.exit(1)

    log.info("\u52a0\u8f7d\u5d4c\u5165\u6a21\u578b: %s", EMBEDDING_MODEL)
    if not os.path.exists(EMBEDDING_MODEL):
        log.warning("\u6a21\u578b\u8def\u5f84\u4e0d\u5b58\u5728\uff0c\u5c1d\u8bd5\u4ece HuggingFace Hub \u4e0b\u8f7d BAAI/bge-base-zh-v1.5 ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    total = len(texts)
    log.info("\u5d4c\u5165 %d \u4e2a\u6587\u6863\u5757 (batch_size=%d) ...", total, BATCH_SIZE)

    all_embeddings = []
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embs)
        done = min(start + BATCH_SIZE, total)
        log.info("  [%d/%d]", done, total)

    embeddings = np.vstack(all_embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    log.info("FAISS \u7d22\u5f15\u5df2\u4fdd\u5b58: %s", FAISS_INDEX_PATH)


def save_chunks_and_meta(chunks: list[dict], modes: list[str]) -> None:
    """Persist chunks and build metadata."""
    os.makedirs(INDEX_DIR, exist_ok=True)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)

    meta = {
        "num_chunks": len(chunks),
        "modes": modes,
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": EMBEDDING_MODEL if "dense" in modes else None,
    }
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="\u6784\u5efa Ascend C \u77e5\u8bc6\u5e93: Markdown -> \u5206\u5757 -> \u7d22\u5f15",
    )
    parser.add_argument("--dense", action="store_true",
                        help="\u540c\u65f6\u6784\u5efa FAISS \u7a20\u5bc6\u7d22\u5f15 (\u9700\u8981 GPU \u548c\u6a21\u578b)")
    parser.add_argument("--md", type=str, default=None,
                        help="\u81ea\u5b9a\u4e49 Markdown \u6587\u4ef6\u8def\u5f84 (\u9ed8\u8ba4\u4f7f\u7528 MinerU \u89e3\u6790\u7ed3\u679c)")
    parser.add_argument("-v", "--verbose", action="store_true", help="\u8be6\u7ec6\u65e5\u5fd7")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    md_path = args.md or MD_PATH

    log.info("=" * 60)
    log.info("Ascend C \u77e5\u8bc6\u5e93\u6784\u5efa\u5de5\u5177")
    log.info("=" * 60)

    log.info("[1/3] \u8bfb\u53d6 Markdown: %s", md_path)
    text = read_markdown(md_path)
    log.info("  \u603b\u5b57\u7b26\u6570: %s", f"{len(text):,}")

    log.info("[2/3] \u5206\u5757\u5904\u7406 ...")
    sections = split_by_headers(text)
    log.info("  \u539f\u59cb\u7ae0\u8282\u6570: %d", len(sections))
    sections = merge_small_sections(sections, CHUNK_MIN_CHARS)
    log.info("  \u5408\u5e76\u540e: %d", len(sections))
    chunks = build_chunks(sections)
    log.info("  \u6700\u7ec8\u6587\u6863\u5757\u6570: %d", len(chunks))
    avg_len = sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)
    log.info("  \u5e73\u5747\u5757\u957f\u5ea6: %.0f chars", avg_len)

    log.info("[3/3] \u6784\u5efa\u7d22\u5f15 ...")
    modes = ["sparse"]
    os.makedirs(INDEX_DIR, exist_ok=True)
    build_tfidf_index(chunks)

    if args.dense:
        modes.append("dense")
        build_dense_index(chunks)

    save_chunks_and_meta(chunks, modes)

    log.info("=" * 60)
    log.info("\u77e5\u8bc6\u5e93\u6784\u5efa\u5b8c\u6210!")
    log.info("  \u6587\u6863\u5757\u6570: %d", len(chunks))
    log.info("  \u7d22\u5f15\u6a21\u5f0f: %s", ", ".join(modes))
    log.info("  \u5b58\u50a8\u8def\u5f84: %s", INDEX_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()

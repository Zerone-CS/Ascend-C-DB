#!/usr/bin/env python3
"""Query the Ascend C knowledge base.

Supports sparse (TF-IDF), dense (FAISS), and hybrid retrieval modes.

Usage:
    python -m knowledge_base.query_kb -q 'Tiling\u5207\u5206\u7b56\u7565'
    python -m knowledge_base.query_kb -q 'DoubleBuffer' -k 10 -v
    python -m knowledge_base.query_kb                          # interactive
"""

import argparse
import json
import logging
import os
import pickle
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer_utils import jieba_tokenizer

import numpy as np

from kb_config import (
    CHUNKS_PATH,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    META_PATH,
    TFIDF_PATH,
)

log = logging.getLogger(__name__)


class KnowledgeBase:
    """Ascend C knowledge base with sparse / dense / hybrid retrieval."""

    def __init__(self, mode: str = "sparse"):
        if mode not in ("sparse", "dense", "hybrid"):
            raise ValueError(f"Unsupported mode: {mode!r}  (sparse | dense | hybrid)")
        self.mode = mode
        self.chunks = None
        self.meta = None
        self._load_chunks()

        if mode in ("sparse", "hybrid"):
            self._load_sparse()
        if mode in ("dense", "hybrid"):
            self._load_dense()

    def _load_chunks(self):
        if not os.path.exists(CHUNKS_PATH):
            log.error("\u77e5\u8bc6\u5e93\u4e0d\u5b58\u5728\uff0c\u8bf7\u5148\u8fd0\u884c: python -m knowledge_base.build_kb")
            raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as fh:
            self.chunks = json.load(fh)
        with open(META_PATH, "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)
        log.info("\u77e5\u8bc6\u5e93\u5df2\u52a0\u8f7d: %d \u4e2a\u6587\u6863\u5757 (\u6784\u5efa\u4e8e %s)",
                 self.meta["num_chunks"], self.meta["build_time"])

    def _load_sparse(self):
        if not os.path.exists(TFIDF_PATH):
            raise FileNotFoundError(f"TF-IDF index not found: {TFIDF_PATH}")
        with open(TFIDF_PATH, "rb") as fh:
            data = pickle.load(fh)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["matrix"]
        log.info("\u2705 TF-IDF \u7d22\u5f15\u5df2\u52a0\u8f7d")

    def _load_dense(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("\u7f3a\u5c11\u4f9d\u8d56: pip install -r requirements-dense.txt")

        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found: {FAISS_INDEX_PATH}\n"
                "\u8bf7\u8fd0\u884c: python -m knowledge_base.build_kb --dense"
            )
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        embed_model_path = self.meta.get("embedding_model") or EMBEDDING_MODEL
        self.embed_model = SentenceTransformer(embed_model_path)
        log.info("\u2705 FAISS \u7d22\u5f15\u5df2\u52a0\u8f7d")

    def search_sparse(self, query: str, top_k: int) -> list[dict]:
        """Retrieve chunks via TF-IDF cosine similarity."""
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "title": chunk["title"],
                "text": chunk["text"],
                "line": chunk["line"],
                "chunk_id": chunk["id"],
                "method": "sparse",
            })
        return results

    def search_dense(self, query: str, top_k: int) -> list[dict]:
        """Retrieve chunks via FAISS inner-product search."""
        query_emb = self.embed_model.encode(
            [query], normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.faiss_index.search(query_emb, top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "title": chunk["title"],
                "text": chunk["text"],
                "line": chunk["line"],
                "chunk_id": chunk["id"],
                "method": "dense",
            })
        return results

    def search_hybrid(self, query: str, top_k: int,
                      sparse_weight: float = 0.4,
                      dense_weight: float = 0.6) -> list[dict]:
        """Combine sparse and dense results with weighted fusion."""
        sparse_res = self.search_sparse(query, top_k * 2)
        dense_res = self.search_dense(query, top_k * 2)

        score_map = {}
        for res in sparse_res:
            cid = res["chunk_id"]
            score_map[cid] = {"sparse": res["score"], "dense": 0.0, "data": res}
        for res in dense_res:
            cid = res["chunk_id"]
            if cid in score_map:
                score_map[cid]["dense"] = res["score"]
            else:
                score_map[cid] = {"sparse": 0.0, "dense": res["score"], "data": res}

        for entry in score_map.values():
            entry["combined"] = sparse_weight * entry["sparse"] + dense_weight * entry["dense"]

        ranked = sorted(score_map.values(), key=lambda x: x["combined"], reverse=True)[:top_k]
        results = []
        for rank, entry in enumerate(ranked, start=1):
            data = entry["data"]
            data["rank"] = rank
            data["score"] = entry["combined"]
            data["method"] = "hybrid"
            results.append(data)
        return results

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Dispatch search to the configured retrieval mode."""
        if self.mode == "sparse":
            return self.search_sparse(query, top_k)
        elif self.mode == "dense":
            return self.search_dense(query, top_k)
        else:
            return self.search_hybrid(query, top_k)


def format_results(results: list[dict], verbose: bool = False) -> str:
    """Format search results for terminal display."""
    if not results:
        return "\u672a\u627e\u5230\u76f8\u5173\u7ed3\u679c"

    parts = []
    for res in results:
        header = f"\n{'=' * 60}"
        header += (f"\n[\u7ed3\u679c {res['rank']}]  "
                   f"\u76f8\u5173\u5ea6: {res['score']:.4f}  |  "
                   f"\u6807\u9898: {res['title']}  |  "
                   f"\u884c\u53f7: {res['line']}")
        header += f"\n{'-' * 60}"
        text = res["text"]
        if not verbose and len(text) > 600:
            text = text[:600] + "\n... (\u4f7f\u7528 -v \u67e5\u770b\u5b8c\u6574\u5185\u5bb9)"
        parts.append(header + "\n" + text)
    parts.append("\n" + "=" * 60)
    return "\n".join(parts)


def interactive_mode(kb: KnowledgeBase, top_k: int, verbose: bool) -> None:
    """Run an interactive query loop in the terminal."""
    print(f"\n\u8fdb\u5165\u4ea4\u4e92\u67e5\u8be2\u6a21\u5f0f (\u6a21\u5f0f: {kb.mode}, Top-K: {top_k})")
    print("\u8f93\u5165 'quit' \u6216 'exit' \u9000\u51fa\n")

    while True:
        try:
            query = input("\U0001f50d \u8bf7\u8f93\u5165\u67e5\u8be2: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\u518d\u89c1!")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            print("\u518d\u89c1!")
            break
        results = kb.search(query, top_k)
        print(format_results(results, verbose))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Ascend C \u77e5\u8bc6\u5e93\u67e5\u8be2\u5de5\u5177",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\u793a\u4f8b:\n"
            "  python -m knowledge_base.query_kb -q '\u5982\u4f55\u5b9e\u73b0Ascend C\u77e2\u91cf\u7b97\u5b50'\n"
            "  python -m knowledge_base.query_kb -q 'Tiling\u5207\u5206\u7b56\u7565' -k 10 -v\n"
            "  python -m knowledge_base.query_kb -q 'DoubleBuffer' --mode sparse\n"
            "  python -m knowledge_base.query_kb   # \u4ea4\u4e92\u6a21\u5f0f\n"
        ),
    )
    parser.add_argument("-q", "--query", type=str, default=None, help="\u67e5\u8be2\u5185\u5bb9")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help="\u8fd4\u56de\u7ed3\u679c\u6570 (\u9ed8\u8ba4: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="\u663e\u793a\u5b8c\u6574\u6587\u672c")
    parser.add_argument("-m", "--mode", choices=["sparse", "dense", "hybrid"], default="sparse",
                        help="\u68c0\u7d22\u6a21\u5f0f (\u9ed8\u8ba4: %(default)s)")
    parser.add_argument("--json", action="store_true", help="\u4ee5 JSON \u683c\u5f0f\u8f93\u51fa\u7ed3\u679c")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    kb = KnowledgeBase(mode=args.mode)

    if args.query:
        results = kb.search(args.query, args.top_k)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(format_results(results, args.verbose))
    else:
        interactive_mode(kb, args.top_k, args.verbose)


if __name__ == "__main__":
    main()

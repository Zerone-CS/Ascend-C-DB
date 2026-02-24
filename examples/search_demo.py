#!/usr/bin/env python3
"""Demo: search the Ascend C knowledge base."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge_base import KnowledgeBase


def main():
    kb = KnowledgeBase(mode="sparse")

    queries = [
        "DoubleBuffer \u5b9e\u73b0\u65b9\u6cd5",
        "Tiling \u591a\u6838\u5207\u5206\u7b56\u7565",
        "Matmul \u77e9\u9635\u4e58 API",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print("\u67e5\u8be2: {}".format(query))
        print("=" * 60)

        results = kb.search(query, top_k=3)
        for r in results:
            print("  [{:.4f}] {}".format(r["score"], r["title"]))
            print("           {}...".format(r["text"][:100]))
            print()


if __name__ == "__main__":
    main()

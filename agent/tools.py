"""Tool definitions for the Ascend C Agent.

Provides:
- search_ascend_docs: search the knowledge base
- get_tool_definitions: OpenAI function-calling compatible schema
"""

import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge_base.query_kb import KnowledgeBase

_kb_instance: Optional[KnowledgeBase] = None


def _get_kb(mode: str = "sparse") -> KnowledgeBase:
    """Lazy-init singleton KnowledgeBase."""
    global _kb_instance
    if _kb_instance is None or _kb_instance.mode != mode:
        _kb_instance = KnowledgeBase(mode=mode)
    return _kb_instance


def search_ascend_docs(
    query: str,
    top_k: int = 5,
    mode: str = "sparse",
) -> list[dict]:
    """Search the Ascend C knowledge base and return relevant document chunks.

    Args:
        query: Search query in Chinese or English.
        top_k: Number of results to return.
        mode: Retrieval mode (sparse / dense / hybrid).

    Returns:
        List of dicts with keys: title, text, score, line.
    """
    kb = _get_kb(mode)
    results = kb.search(query, top_k=top_k)
    return [
        {
            "title": r["title"],
            "text": r["text"],
            "score": round(r["score"], 4),
            "line": r["line"],
        }
        for r in results
    ]


def search_ascend_docs_formatted(
    query: str,
    top_k: int = 5,
    mode: str = "sparse",
) -> str:
    """Search and return results as a formatted string for LLM context."""
    results = search_ascend_docs(query, top_k, mode)
    if not results:
        return f"\u672a\u627e\u5230\u4e0e '{query}' \u76f8\u5173\u7684\u6587\u6863\u5185\u5bb9\u3002"

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"--- \u6587\u6863\u7247\u6bb5 {i} (\u76f8\u5173\u5ea6: {r['score']}) ---\n"
            f"\u6807\u9898: {r['title']}\n"
            f"{r['text']}"
        )
    return "\n\n".join(parts)


def get_tool_definitions() -> list[dict]:
    """Return OpenAI function-calling compatible tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_ascend_docs",
                "description": (
                    "\u641c\u7d22\u6607\u817e Ascend C \u7b97\u5b50\u5f00\u53d1\u5b98\u65b9\u6587\u6863\u3002"
                    "\u5f53\u9700\u8981\u67e5\u8be2 API \u7528\u6cd5\u3001\u7f16\u7a0b\u8303\u5f0f\u3001"
                    "Tiling \u7b56\u7565\u3001DoubleBuffer\u3001\u6570\u636e\u7c7b\u578b\u3001\u6846\u67b6\u9002\u914d\u7b49\u5185\u5bb9\u65f6\u8c03\u7528\u3002"
                    "\u6bcf\u6b21\u5199\u7b97\u5b50\u4ee3\u7801\u524d\u5fc5\u987b\u5148\u8c03\u7528\u6b64\u5de5\u5177\u83b7\u53d6\u51c6\u786e\u4fe1\u606f\u3002"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "\u641c\u7d22\u67e5\u8be2\uff0c\u4f8b\u5982 '\u77e2\u91cf\u52a0\u6cd5\u7b97\u5b50\u5b9e\u73b0' \u6216 'DoubleBuffer \u914d\u7f6e'",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "\u8fd4\u56de\u7ed3\u679c\u6570\u91cf\uff0c\u9ed8\u8ba4 5",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def handle_tool_call(name: str, arguments: dict) -> str:
    """Dispatch a tool call by name and return the result as a string."""
    if name == "search_ascend_docs":
        return search_ascend_docs_formatted(
            query=arguments["query"],
            top_k=arguments.get("top_k", 5),
        )
    raise ValueError(f"Unknown tool: {name}")

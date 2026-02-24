#!/usr/bin/env python3
"""Demo: integrate the search tool into your own Agent framework.

This example shows how to use the tool definitions and search functions
with your own LLM / Agent setup (LangChain, AutoGen, custom, etc.).
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.tools import (
    get_tool_definitions,
    handle_tool_call,
    search_ascend_docs,
    search_ascend_docs_formatted,
)


def main():
    # 1. Get OpenAI function-calling compatible tool definitions
    tools = get_tool_definitions()
    print("Tool definitions (send this to your LLM):")
    print(json.dumps(tools, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)

    # 2. Direct search - returns structured results
    results = search_ascend_docs("DoubleBuffer", top_k=3)
    print("\nStructured results:")
    for r in results:
        print("  [{score:.4f}] {title}".format(**r))

    print("\n" + "=" * 60)

    # 3. Formatted search - returns a string ready for LLM context
    context = search_ascend_docs_formatted("Tiling \u5c3e\u5757\u5904\u7406", top_k=3)
    print("\nFormatted context (inject this into your prompt):")
    print(context[:500] + "...")

    print("\n" + "=" * 60)

    # 4. Handle a tool call from LLM (dispatch by name)
    result = handle_tool_call("search_ascend_docs", {"query": "\u6838\u51fd\u6570\u7f16\u5199", "top_k": 2})
    print("\nTool call result:")
    print(result[:300] + "...")


if __name__ == "__main__":
    main()

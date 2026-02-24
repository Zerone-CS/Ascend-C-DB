#!/usr/bin/env python3
"""Ascend C Knowledge Base & Agent - top-level entry point.

Usage:
    python main.py build [--dense] [--md PATH]   # 构建索引
    python main.py query -q '...' [-k N] [-v]    # 单次查询
    python main.py query                          # 交互模式
    python main.py agent -q '...'                 # Agent 单次提问
    python main.py agent                          # Agent 交互模式
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    usage = (
        "Usage: python main.py <command> [options]\n"
        "\n"
        "Commands:\n"
        "  build   \u6784\u5efa\u77e5\u8bc6\u5e93\u7d22\u5f15\n"
        "  query   \u67e5\u8be2\u77e5\u8bc6\u5e93\n"
        "  agent   \u542f\u52a8 RAG \u7b97\u5b50\u5f00\u53d1\u52a9\u624b\n"
        "\n"
        "Examples:\n"
        "  python main.py build\n"
        "  python main.py query -q 'Tiling\u5207\u5206'\n"
        "  python main.py agent -q '\u5e2e\u6211\u5199\u4e00\u4e2a Add \u7b97\u5b50'\n"
        "  python main.py agent              # \u4ea4\u4e92\u6a21\u5f0f\n"
    )

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(usage)
        sys.exit(0)

    command = sys.argv.pop(1)

    if command == "build":
        from knowledge_base.build_kb import main as build_main
        build_main()
    elif command == "query":
        from knowledge_base.query_kb import main as query_main
        query_main()
    elif command == "agent":
        from agent.ascend_agent import main as agent_main
        agent_main()
    else:
        print(f"Unknown command: {command}\n")
        print(usage)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""AscendMind RAG Agent - uses tool calling to search docs before writing operators.

Supports any OpenAI-compatible API (OpenAI, vLLM, Ollama, etc.).

Usage:
    python -m agent.ascend_agent --question '帮我写一个 Add 算子'
    python -m agent.ascend_agent   # interactive mode
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from agent.prompts import RAG_CONTEXT_TEMPLATE, SYSTEM_PROMPT
from agent.tools import (
    get_tool_definitions,
    handle_tool_call,
    search_ascend_docs_formatted,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"
MAX_TOOL_ROUNDS = 5


class AscendCAgent:
    """RAG-enhanced Agent for Ascend C operator development.

    Wraps any OpenAI-compatible chat API with automatic tool calling
    to search the AscendMind knowledge base.

    Args:
        api_key:  API key (or set OPENAI_API_KEY / ASCEND_AGENT_API_KEY env var).
        base_url: API base URL (or set ASCEND_AGENT_BASE_URL env var).
        model:    Model name  (or set ASCEND_AGENT_MODEL env var).
        mode:     RAG retrieval mode: 'tool_call' | 'pre_retrieve'.
                  - tool_call:    let the LLM decide when to search (function calling).
                  - pre_retrieve: always search before answering (simpler, works with any model).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        mode: str = "tool_call",
    ):
        self.api_key = (
            api_key
            or os.environ.get("ASCEND_AGENT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.base_url = (
            base_url
            or os.environ.get("ASCEND_AGENT_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = (
            model
            or os.environ.get("ASCEND_AGENT_MODEL")
            or DEFAULT_MODEL
        )
        self.mode = mode
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        log.info("模型: %s  基址: %s  模式: %s", self.model, self.base_url, self.mode)

    def _call_api(self, extra_kwargs: Optional[dict] = None) -> dict:
        """Send a chat completion request."""
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0.3,
        }
        if extra_kwargs:
            payload.update(extra_kwargs)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    def _tool_call_loop(self, question: str) -> str:
        """Let the LLM decide when to call search_ascend_docs."""
        self.messages.append({"role": "user", "content": question})

        for round_idx in range(MAX_TOOL_ROUNDS):
            log.info("─ Tool-call 轮次 %d/%d", round_idx + 1, MAX_TOOL_ROUNDS)

            result = self._call_api({"tools": get_tool_definitions()})
            choice = result["choices"][0]
            msg = choice["message"]

            if msg.get("tool_calls"):
                self.messages.append(msg)
                for tc in msg["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    log.info("  └ 调用工具: %s(%s)", fn_name, fn_args)

                    tool_result = handle_tool_call(fn_name, fn_args)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })
            else:
                content = msg.get("content", "")
                self.messages.append({"role": "assistant", "content": content})
                return content

        return self.messages[-1].get("content", "达到最大工具调用次数")

    def _pre_retrieve(self, question: str) -> str:
        """Always retrieve first, then answer (no function-calling needed)."""
        log.info("─ Pre-retrieve 模式: 检索中 ...")
        context = search_ascend_docs_formatted(question, top_k=5)
        log.info("  └ 检索完成, 上下文长度: %d 字符", len(context))

        augmented = RAG_CONTEXT_TEMPLATE.format(context=context)
        self.messages.append({"role": "user", "content": augmented + "\n\n" + question})

        result = self._call_api()
        content = result["choices"][0]["message"].get("content", "")
        self.messages.append({"role": "assistant", "content": content})
        return content

    def chat(self, question: str) -> str:
        """Send a question and get a RAG-enhanced answer.

        Args:
            question: User question about Ascend C operator development.

        Returns:
            The assistant's response text.
        """
        if self.mode == "tool_call":
            return self._tool_call_loop(question)
        else:
            return self._pre_retrieve(question)

    def reset(self):
        """Clear conversation history, keeping only the system prompt."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def interactive_mode(agent: AscendCAgent):
    """Run an interactive chat session."""
    print("\n\U0001f680 Ascend C 算子开发助手 (\u8f93\u5165 quit 退\u51fa, reset 清\u7a7a\u5bf9\u8bdd)\n")
    while True:
        try:
            question = input("\U0001f9d1 \u4f60: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\u518d\u89c1!")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\u518d\u89c1!")
            break
        if question.lower() == "reset":
            agent.reset()
            print("\u2705 \u5bf9\u8bdd\u5df2\u91cd\u7f6e\n")
            continue

        try:
            answer = agent.chat(question)
            print(f"\n\U0001f916 \u52a9\u624b:\n{answer}\n")
        except httpx.HTTPStatusError as exc:
            print(f"\n\u274c API \u9519\u8bef: {exc.response.status_code} {exc.response.text}\n")
        except httpx.ConnectError:
            print(f"\n\u274c \u65e0\u6cd5\u8fde\u63a5 API: {agent.base_url}")
            print("\u8bf7\u68c0\u67e5 ASCEND_AGENT_BASE_URL \u548c\u7f51\u7edc\u8fde\u63a5\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ascend C \u7b97\u5b50\u5f00\u53d1 RAG Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\u73af\u5883\u53d8\u91cf:\n"
            "  ASCEND_AGENT_API_KEY    API \u5bc6\u94a5 (\u6216 OPENAI_API_KEY)\n"
            "  ASCEND_AGENT_BASE_URL   API \u57fa\u5740 (\u9ed8\u8ba4: https://api.openai.com/v1)\n"
            "  ASCEND_AGENT_MODEL      \u6a21\u578b\u540d\u79f0 (\u9ed8\u8ba4: gpt-4o)\n"
            "\n"
            "\u793a\u4f8b:\n"
            "  python -m agent.ascend_agent -q '\u5e2e\u6211\u5199\u4e00\u4e2a Add \u7b97\u5b50'\n"
            "  python -m agent.ascend_agent --mode pre_retrieve\n"
            "  ASCEND_AGENT_BASE_URL=http://localhost:8000/v1 python -m agent.ascend_agent\n"
        ),
    )
    parser.add_argument("-q", "--question", type=str, default=None,
                        help="\u5355\u6b21\u63d0\u95ee")
    parser.add_argument("--mode", choices=["tool_call", "pre_retrieve"],
                        default="tool_call",
                        help="RAG \u6a21\u5f0f: tool_call (\u6a21\u578b\u81ea\u4e3b\u68c0\u7d22) | pre_retrieve (\u5148\u68c0\u7d22\u518d\u56de\u7b54)")
    parser.add_argument("--base-url", type=str, default=None, help="API base URL")
    parser.add_argument("--model", type=str, default=None, help="\u6a21\u578b\u540d\u79f0")
    parser.add_argument("--api-key", type=str, default=None, help="API \u5bc6\u94a5")
    parser.add_argument("-v", "--verbose", action="store_true", help="\u8be6\u7ec6\u65e5\u5fd7")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    agent = AscendCAgent(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        mode=args.mode,
    )

    if args.question:
        try:
            answer = agent.chat(args.question)
            print(answer)
        except httpx.HTTPStatusError as exc:
            log.error("API \u9519\u8bef: %s %s", exc.response.status_code, exc.response.text)
            sys.exit(1)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()

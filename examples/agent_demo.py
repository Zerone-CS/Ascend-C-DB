#!/usr/bin/env python3
"""Demo: use the RAG Agent to generate Ascend C operator code.

Requires:
    export ASCEND_AGENT_API_KEY='your-key'
    export ASCEND_AGENT_BASE_URL='https://api.deepseek.com'  # or other
    export ASCEND_AGENT_MODEL='deepseek-chat'
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import AscendCAgent


def main():
    agent = AscendCAgent(mode="tool_call")

    questions = [
        "\u5e2e\u6211\u5199\u4e00\u4e2a\u652f\u6301 DoubleBuffer \u7684 Add \u7b97\u5b50",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print("\u63d0\u95ee: {}".format(question))
        print("=" * 60)

        answer = agent.chat(question)
        print(answer)
        agent.reset()


if __name__ == "__main__":
    main()

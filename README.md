<div align="center">

# AscendMind

**RAG-enhanced knowledge base & intelligent agent for Huawei Ascend C operator development**

AscendMind — 基于检索增强生成 (RAG) 的昇腾 Ascend C 算子开发知识库与智能助手

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)](.github/workflows/ci.yml)

[English](#features) · [中文说明](#功能特性) · [Quick Start](#quick-start) · [Agent](#agent) · [API](#python-api) · [Contributing](CONTRIBUTING.md)

</div>

---

## Features

- 📚 **Knowledge Base** — 5900+ chunks from the official CANN 8.2.RC1 Ascend C Operator Development Guide
- 🔍 **3 Retrieval Modes** — Sparse (TF-IDF), Dense (FAISS + BGE), Hybrid
- 🤖 **RAG Agent** — AI assistant that searches docs before writing operator code
- 🔧 **Tool Integration** — OpenAI function-calling compatible, plug into any agent framework
- 🌍 **Multi-backend** — OpenAI, DeepSeek, vLLM, Ollama, and any compatible API

## 功能特性

- 📚 **知识库** — 基于 CANN 8.2.RC1 官方文档构建，5900+ 文档块
- 🔍 **三种检索模式** — 稀疏 (TF-IDF)、稠密 (FAISS + BGE)、混合
- 🤖 **RAG Agent** — 写算子前自动检索文档，确保 API 用法准确
- 🔧 **工具集成** — 兼容 OpenAI function-calling，可接入任意 Agent 框架
- 🌍 **多后端** — 支持 OpenAI、DeepSeek、vLLM、Ollama 等

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     User / Your App                        │
└──────────────────────────┬─────────────────────────────────┘
                          │
            ┌───────────┴───────────┐
            │   main.py (CLI)       │
            │   build | query | agent│
            └──────┬────────┬──────┘
                   │        │
        ┌────────┴─┐  ┌───┴────────┐
        │knowledge │  │  agent/     │
        │_base/    │  │            │
        │          │  │ prompts.py │
        │ build_kb │  │ tools.py   │
        │ query_kb │  │ ascend_    │
        │ kb_config│  │  agent.py  │
        └────┬─────┘  └────┬───────┘
             │              │
        ┌────┴─────┐     ┌─┴──────────┐
        │index_    │     │OpenAI-     │
        │store/    │     │compatible  │
        │          │     │API         │
        │chunks    │     │(DeepSeek,  │
        │tfidf     │     │ vLLM, etc.)│
        │faiss     │     └────────────┘
        └──────────┘
```

## Project Structure

```
ascend-mind/
├── main.py                    # CLI: python main.py build|query|agent
├── pyproject.toml             # Project metadata & dependencies
├── knowledge_base/            # 知识库模块
│   ├── kb_config.py           # Configuration (env-var overridable)
│   ├── tokenizer_utils.py     # jieba Chinese tokenizer
│   ├── build_kb.py            # Build index: Markdown -> chunks -> TF-IDF/FAISS
│   ├── query_kb.py            # KnowledgeBase class & query engine
│   └── index_store/           # Pre-built index (chunks.json + meta.json)
├── agent/                     # RAG Agent 模块
│   ├── prompts.py             # System prompts for Ascend C expert
│   ├── tools.py               # OpenAI function-calling compatible tools
│   └── ascend_agent.py        # Agent with tool_call / pre_retrieve modes
├── data/                      # Source Markdown (user-provided)
├── examples/                  # Usage examples
├── tests/                     # Unit tests
└── .github/                   # CI & issue templates
```

## Quick Start

### Install

```bash
git clone https://github.com/your-org/ascend-mind.git
cd ascend-mind

# Core (sparse retrieval)
pip install -e .

# With Agent support
pip install -e ".[agent]"

# With dense retrieval (FAISS + BGE)
pip install -e ".[dense]"

# Everything
pip install -e ".[agent,dense,dev]"
```

### Prepare Data

The knowledge base is built from a Markdown file (parsed from the official CANN PDF using [MinerU](https://github.com/opendatalab/MinerU)).

```bash
# Option 1: Use your own Markdown
cp your-document.md data/ascend_c_guide.md

# Option 2: Parse from PDF with MinerU
pip install mineru
mineru -p "CANN_Guide.pdf" -o output/
cp output/auto/*.md data/ascend_c_guide.md
```

### Build Index

```bash
# Sparse only (fast, no GPU)
python main.py build

# Sparse + Dense (needs GPU + BGE model)
python main.py build --dense

# Custom Markdown path
python main.py build --md /path/to/your/doc.md
```

### Search

```bash
# Single query
python main.py query -q 'DoubleBuffer 实现方法'

# More results + full text
python main.py query -q 'Tiling 切分策略' -k 10 -v

# JSON output (for integration)
python main.py query -q 'Matmul API' --json

# Interactive mode
python main.py query
```

## Agent

The RAG Agent automatically searches the knowledge base before writing operator code.

### Setup

```bash
# Configure API (supports any OpenAI-compatible endpoint)
export ASCEND_AGENT_API_KEY='your-api-key'
export ASCEND_AGENT_BASE_URL='https://api.deepseek.com'   # or openai / vllm / ollama
export ASCEND_AGENT_MODEL='deepseek-chat'
```

### Usage

```bash
# Single question
python main.py agent -q '帮我写一个支持 DoubleBuffer 的 Add 算子'

# Interactive mode
python main.py agent

# pre_retrieve mode (works with any model, no function-calling needed)
python main.py agent --mode pre_retrieve
```

### RAG Modes

| Mode | How It Works | Best For |
|------|-------------|----------|
| `tool_call` | LLM decides when & what to search | GPT-4o, DeepSeek, Qwen (function-calling models) |
| `pre_retrieve` | Always search Top-5 before answering | Any model, simpler setup |

### Supported Backends

```bash
# OpenAI
ASCEND_AGENT_BASE_URL=https://api.openai.com/v1

# DeepSeek
ASCEND_AGENT_BASE_URL=https://api.deepseek.com

# vLLM (local)
ASCEND_AGENT_BASE_URL=http://localhost:8000/v1

# Ollama (local)
ASCEND_AGENT_BASE_URL=http://localhost:11434/v1
```

## Python API

### Knowledge Base

```python
from knowledge_base import KnowledgeBase

kb = KnowledgeBase(mode="sparse")  # sparse | dense | hybrid
results = kb.search("矢量编程 DoubleBuffer", top_k=5)

for r in results:
    print(f"[{r['score']:.4f}] {r['title']}")
```

### Agent

```python
from agent import AscendCAgent

agent = AscendCAgent(
    api_key="your-key",
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
)
answer = agent.chat("帮我写一个 Softmax 算子")
print(answer)
```

### Tool Integration (Bring Your Own Agent)

```python
from agent.tools import get_tool_definitions, handle_tool_call

# Get OpenAI function-calling schema
tools = get_tool_definitions()

# Handle LLM's tool_call response
result = handle_tool_call("search_ascend_docs", {"query": "Tiling 尾块处理"})
```

See [`examples/`](examples/) for more usage patterns.

## Configuration

All settings are overridable via environment variables:

| Variable | Description | Default |
|----------|-------------|--------|
| `ASCEND_AGENT_API_KEY` | LLM API key | - |
| `ASCEND_AGENT_BASE_URL` | API endpoint | `https://api.openai.com/v1` |
| `ASCEND_AGENT_MODEL` | Model name | `gpt-4o` |
| `ASCEND_KB_MD_PATH` | Source Markdown path | `data/ascend_c_guide.md` |
| `ASCEND_KB_EMBED_MODEL` | Embedding model | `BAAI/bge-base-zh-v1.5` |
| `ASCEND_KB_CHUNK_MAX` | Max chunk size | `1500` |
| `ASCEND_KB_CHUNK_OVERLAP` | Chunk overlap | `200` |
| `ASCEND_KB_TOP_K` | Default top-k results | `5` |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,agent]"

# Run tests
python -m pytest tests/ -v

# Lint
ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgements

- [CANN](https://www.hiascend.com/software/cann) — Huawei Ascend Computing Architecture for Neural Networks
- [MinerU](https://github.com/opendatalab/MinerU) — PDF to Markdown conversion
- [jieba](https://github.com/fxsjy/jieba) — Chinese text segmentation
- [BGE](https://huggingface.co/BAAI/bge-base-zh-v1.5) — Chinese text embeddings

## License

[Apache License 2.0](LICENSE)

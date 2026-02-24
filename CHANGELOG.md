# Changelog

## [0.1.0] - 2025-02-24

### Added
- Knowledge base: PDF -> Markdown -> chunking -> TF-IDF/FAISS index
- Sparse (TF-IDF + jieba), dense (FAISS + BGE), and hybrid retrieval modes
- RAG Agent with tool_call and pre_retrieve modes
- OpenAI-compatible API support (OpenAI, DeepSeek, vLLM, Ollama, etc.)
- CLI entry point: `python main.py build|query|agent`
- Python API: `KnowledgeBase` class and `AscendCAgent` class

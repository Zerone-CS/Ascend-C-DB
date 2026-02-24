"""Ascend C Knowledge Base package."""

try:
    from knowledge_base.query_kb import KnowledgeBase
except ImportError:
    pass

__all__ = ["KnowledgeBase"]

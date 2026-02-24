"""Chinese tokenizer utilities based on jieba."""

_jieba = None

STOPWORDS = frozenset({
    "\u7684", "\u4e86", "\u5728", "\u662f", "\u6211", "\u6709", "\u548c", "\u5c31", "\u4e0d", "\u4eba", "\u90fd",
    "\u4e00", "\u4e00\u4e2a", "\u4e0a", "\u4e5f", "\u5f88", "\u5230", "\u8bf4", "\u8981", "\u53bb", "\u4f60",
    "\u4f1a", "\u7740", "\u6ca1\u6709", "\u770b", "\u597d", "\u81ea\u5df1", "\u8fd9", "\u4ed6", "\u5979", "\u5b83",
})


def _get_jieba():
    global _jieba
    if _jieba is None:
        import jieba as _jb
        _jieba = _jb
    return _jieba


def jieba_tokenizer(text: str) -> list[str]:
    """Tokenize Chinese text using jieba, filtering stopwords and short tokens."""
    tokens = _get_jieba().lcut(text)
    return [t.strip() for t in tokens
            if t.strip() and len(t.strip()) > 1 and t.strip() not in STOPWORDS]

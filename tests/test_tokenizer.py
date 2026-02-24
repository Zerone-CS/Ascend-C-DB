import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge_base.tokenizer_utils import jieba_tokenizer


def test_basic_tokenization():
    tokens = jieba_tokenizer("Ascend C \u7b97\u5b50\u5f00\u53d1\u6307\u5357")
    assert len(tokens) > 0
    assert all(len(t) > 1 for t in tokens)


def test_stopwords_filtered():
    tokens = jieba_tokenizer("\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5")
    assert "\u8fd9" not in tokens
    assert "\u662f" not in tokens
    assert "\u4e00\u4e2a" not in tokens


def test_empty_input():
    tokens = jieba_tokenizer("")
    assert tokens == []

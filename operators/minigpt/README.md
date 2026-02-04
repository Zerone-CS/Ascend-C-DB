# MiniGPT - AscendC NPU 实现

## 架构概览

```
Input Token IDs [batch, seq_len]
       │
       ▼
┌──────────────────┐
│  Token Embedding │  [vocab_size, hidden_dim]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Position Embed  │  [max_seq, hidden_dim]
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Transformer Block × N_layers     │
│  ┌────────────────────────────────────┐  │
│  │  LayerNorm                         │  │
│  │       ↓                            │  │
│  │  Multi-Head Attention              │  │
│  │  (QKV Proj → Attn → Out Proj)     │  │
│  │       ↓                            │  │
│  │  Residual Add                      │  │
│  │       ↓                            │  │
│  │  LayerNorm                         │  │
│  │       ↓                            │  │
│  │  FFN (Linear→GELU→Linear)         │  │
│  │       ↓                            │  │
│  │  Residual Add                      │  │
│  └────────────────────────────────────┘  │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────┐
│    LayerNorm     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   LM Head        │  [hidden_dim, vocab_size]
└────────┬─────────┘
         │
         ▼
   Logits [batch, seq_len, vocab_size]
```

## 超参数 (MiniGPT-Tiny)

| 参数 | 值 |
|------|----|
| vocab_size | 256 (简化为字符级) |
| hidden_dim | 64 |
| n_heads | 4 |
| n_layers | 2 |
| max_seq_len | 32 |
| ffn_dim | 256 (4 × hidden_dim) |

## 算子清单

| 算子 | 用途 |
|------|------|
| `embedding_lookup` | Token + Position Embedding |
| `layernorm_custom` | Layer Normalization |
| `matmul_custom` | QKV/FFN 线性变换 |
| `softmax_custom` | Attention Softmax |
| `gelu_custom` | FFN 激活函数 |
| `add_custom` | 残差连接 |

## 编译运行

```bash
cd operators/minigpt
bash build.sh
python3 tests/test_minigpt.py
```

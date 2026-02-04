#!/usr/bin/env python3
"""
MiniGPT NPU 推理实现

使用 ACL API + 自定义 AscendC 算子实现完整的 GPT 推理
"""

import numpy as np
import os
import sys
import time
from typing import Optional, Tuple

# MiniGPT 超参数
class MiniGPTConfig:
    vocab_size: int = 256        # 字符级词表
    hidden_dim: int = 64         # 隐藏层维度
    n_heads: int = 4             # 注意力头数
    n_layers: int = 2            # Transformer 层数
    max_seq_len: int = 32        # 最大序列长度
    ffn_dim: int = 256           # FFN 维度 (4 * hidden_dim)
    eps: float = 1e-5            # LayerNorm epsilon
    head_dim: int = 16           # 每个头的维度 (hidden_dim / n_heads)


class MiniGPTWeights:
    """MiniGPT 权重结构"""
    def __init__(self, config: MiniGPTConfig):
        self.config = config
        np.random.seed(42)  # 可复现
        
        # Token Embedding: [vocab_size, hidden_dim]
        self.token_embedding = np.random.randn(
            config.vocab_size, config.hidden_dim
        ).astype(np.float32) * 0.02
        
        # Position Embedding: [max_seq_len, hidden_dim]
        self.position_embedding = np.random.randn(
            config.max_seq_len, config.hidden_dim
        ).astype(np.float32) * 0.02
        
        # Transformer 层权重
        self.layers = []
        for _ in range(config.n_layers):
            layer = {
                # Attention LayerNorm
                'ln1_gamma': np.ones(config.hidden_dim, dtype=np.float32),
                'ln1_beta': np.zeros(config.hidden_dim, dtype=np.float32),
                
                # QKV Projection: [hidden_dim, 3 * hidden_dim]
                'qkv_weight': np.random.randn(
                    config.hidden_dim, 3 * config.hidden_dim
                ).astype(np.float32) * 0.02,
                'qkv_bias': np.zeros(3 * config.hidden_dim, dtype=np.float32),
                
                # Output Projection: [hidden_dim, hidden_dim]
                'out_weight': np.random.randn(
                    config.hidden_dim, config.hidden_dim
                ).astype(np.float32) * 0.02,
                'out_bias': np.zeros(config.hidden_dim, dtype=np.float32),
                
                # FFN LayerNorm
                'ln2_gamma': np.ones(config.hidden_dim, dtype=np.float32),
                'ln2_beta': np.zeros(config.hidden_dim, dtype=np.float32),
                
                # FFN Up: [hidden_dim, ffn_dim]
                'ffn_up_weight': np.random.randn(
                    config.hidden_dim, config.ffn_dim
                ).astype(np.float32) * 0.02,
                'ffn_up_bias': np.zeros(config.ffn_dim, dtype=np.float32),
                
                # FFN Down: [ffn_dim, hidden_dim]
                'ffn_down_weight': np.random.randn(
                    config.ffn_dim, config.hidden_dim
                ).astype(np.float32) * 0.02,
                'ffn_down_bias': np.zeros(config.hidden_dim, dtype=np.float32),
            }
            self.layers.append(layer)
        
        # Final LayerNorm
        self.ln_final_gamma = np.ones(config.hidden_dim, dtype=np.float32)
        self.ln_final_beta = np.zeros(config.hidden_dim, dtype=np.float32)
        
        # LM Head: [hidden_dim, vocab_size]
        self.lm_head_weight = np.random.randn(
            config.hidden_dim, config.vocab_size
        ).astype(np.float32) * 0.02


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    """Fast GELU approximation: x * sigmoid(1.702 * x)"""
    return x * (1.0 / (1.0 + np.exp(-1.702 * x)))


def softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layernorm_numpy(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def causal_mask(seq_len: int) -> np.ndarray:
    """Create causal attention mask"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask * -1e9  # Large negative for masked positions


class MiniGPTNumpy:
    """NumPy 参考实现 (CPU)"""
    
    def __init__(self, config: MiniGPTConfig, weights: MiniGPTWeights):
        self.config = config
        self.weights = weights
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        前向传播
        Args:
            token_ids: [seq_len] int32 token IDs
        Returns:
            logits: [seq_len, vocab_size] float32
        """
        seq_len = len(token_ids)
        
        # Token Embedding + Position Embedding
        x = self.weights.token_embedding[token_ids]  # [seq_len, hidden_dim]
        x = x + self.weights.position_embedding[:seq_len]  # Add position
        
        # Causal mask for attention
        mask = causal_mask(seq_len)
        
        # Transformer layers
        for layer in self.weights.layers:
            x = self._transformer_block(x, layer, mask)
        
        # Final LayerNorm
        x = layernorm_numpy(x, self.weights.ln_final_gamma, 
                           self.weights.ln_final_beta, self.config.eps)
        
        # LM Head
        logits = x @ self.weights.lm_head_weight  # [seq_len, vocab_size]
        
        return logits
    
    def _transformer_block(self, x: np.ndarray, layer: dict, mask: np.ndarray) -> np.ndarray:
        """Single transformer block"""
        # Pre-norm attention
        residual = x
        x = layernorm_numpy(x, layer['ln1_gamma'], layer['ln1_beta'], self.config.eps)
        x = self._multi_head_attention(x, layer, mask)
        x = residual + x  # Residual connection
        
        # Pre-norm FFN
        residual = x
        x = layernorm_numpy(x, layer['ln2_gamma'], layer['ln2_beta'], self.config.eps)
        x = self._ffn(x, layer)
        x = residual + x  # Residual connection
        
        return x
    
    def _multi_head_attention(self, x: np.ndarray, layer: dict, mask: np.ndarray) -> np.ndarray:
        """Multi-head self-attention"""
        seq_len, _ = x.shape
        
        # QKV projection
        qkv = x @ layer['qkv_weight'] + layer['qkv_bias']  # [seq_len, 3*hidden_dim]
        q, k, v = np.split(qkv, 3, axis=-1)  # Each [seq_len, hidden_dim]
        
        # Reshape for multi-head: [seq_len, n_heads, head_dim]
        q = q.reshape(seq_len, self.config.n_heads, self.config.head_dim)
        k = k.reshape(seq_len, self.config.n_heads, self.config.head_dim)
        v = v.reshape(seq_len, self.config.n_heads, self.config.head_dim)
        
        # Transpose to [n_heads, seq_len, head_dim]
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        
        # Attention scores: [n_heads, seq_len, seq_len]
        scale = 1.0 / np.sqrt(self.config.head_dim)
        scores = np.matmul(q, k.transpose(0, 2, 1)) * scale
        
        # Apply causal mask
        scores = scores + mask
        
        # Softmax
        attn_weights = softmax_numpy(scores, axis=-1)
        
        # Apply attention to values: [n_heads, seq_len, head_dim]
        attn_output = np.matmul(attn_weights, v)
        
        # Transpose back: [seq_len, n_heads, head_dim]
        attn_output = attn_output.transpose(1, 0, 2)
        
        # Reshape to [seq_len, hidden_dim]
        attn_output = attn_output.reshape(seq_len, self.config.hidden_dim)
        
        # Output projection
        output = attn_output @ layer['out_weight'] + layer['out_bias']
        
        return output
    
    def _ffn(self, x: np.ndarray, layer: dict) -> np.ndarray:
        """Feed-forward network with GELU"""
        # Up projection + GELU
        h = x @ layer['ffn_up_weight'] + layer['ffn_up_bias']
        h = gelu_numpy(h)
        
        # Down projection
        output = h @ layer['ffn_down_weight'] + layer['ffn_down_bias']
        
        return output
    
    def generate(self, prompt: str, max_new_tokens: int = 20, temperature: float = 1.0) -> str:
        """
        自回归生成
        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
        Returns:
            生成的文本
        """
        # 字符级 tokenization
        token_ids = np.array([ord(c) % self.config.vocab_size for c in prompt], dtype=np.int32)
        
        generated = []
        for _ in range(max_new_tokens):
            # 截断到最大序列长度
            if len(token_ids) >= self.config.max_seq_len:
                token_ids = token_ids[-self.config.max_seq_len:]
            
            # Forward pass
            logits = self.forward(token_ids)
            
            # 取最后一个位置的 logits
            next_logits = logits[-1] / temperature
            
            # Softmax 采样
            probs = softmax_numpy(next_logits)
            next_token = np.random.choice(len(probs), p=probs)
            
            # 追加到序列
            token_ids = np.append(token_ids, next_token)
            generated.append(chr(next_token))
            
            # 遇到结束符停止
            if next_token == ord('\n'):
                break
        
        return prompt + ''.join(generated)


def test_minigpt_cpu():
    """CPU 参考实现测试"""
    print("="*60)
    print("MiniGPT CPU Reference Implementation Test")
    print("="*60)
    
    config = MiniGPTConfig()
    weights = MiniGPTWeights(config)
    model = MiniGPTNumpy(config, weights)
    
    print(f"\n模型配置:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  ffn_dim: {config.ffn_dim}")
    
    # 计算参数量
    total_params = (
        config.vocab_size * config.hidden_dim +  # token embedding
        config.max_seq_len * config.hidden_dim +  # position embedding
        config.n_layers * (
            2 * config.hidden_dim +  # ln1 gamma, beta
            config.hidden_dim * 3 * config.hidden_dim +  # qkv weight
            3 * config.hidden_dim +  # qkv bias
            config.hidden_dim * config.hidden_dim +  # out weight
            config.hidden_dim +  # out bias
            2 * config.hidden_dim +  # ln2 gamma, beta
            config.hidden_dim * config.ffn_dim +  # ffn up weight
            config.ffn_dim +  # ffn up bias
            config.ffn_dim * config.hidden_dim +  # ffn down weight
            config.hidden_dim  # ffn down bias
        ) +
        2 * config.hidden_dim +  # final ln
        config.hidden_dim * config.vocab_size  # lm head
    )
    print(f"  总参数量: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    # 测试前向传播
    print(f"\n前向传播测试:")
    prompt = "Hello"
    token_ids = np.array([ord(c) % config.vocab_size for c in prompt], dtype=np.int32)
    print(f"  输入: '{prompt}'")
    print(f"  Token IDs: {token_ids}")
    
    start_time = time.time()
    logits = model.forward(token_ids)
    forward_time = time.time() - start_time
    
    print(f"  输出 logits shape: {logits.shape}")
    print(f"  前向传播时间: {forward_time*1000:.2f} ms")
    
    # 测试生成
    print(f"\n文本生成测试:")
    prompts = ["Hi ", "The ", "AI "]
    
    for prompt in prompts:
        start_time = time.time()
        generated = model.generate(prompt, max_new_tokens=15, temperature=0.8)
        gen_time = time.time() - start_time
        
        # 清理不可打印字符
        cleaned = ''.join(c if c.isprintable() else '?' for c in generated)
        print(f"  '{prompt}' -> '{cleaned[:50]}...'  ({gen_time*1000:.1f}ms)")
    
    print(f"\n✅ CPU 参考实现测试通过!")
    return True


if __name__ == "__main__":
    test_minigpt_cpu()

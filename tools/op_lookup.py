#!/usr/bin/env python3
"""
算子快速查询工具 - 分类 + API + 模板建议

用法:
    python op_lookup.py <算子名>
    python op_lookup.py softmax
    python op_lookup.py gelu
    python op_lookup.py list        # 列出所有支持的算子
"""

import sys

# 算子分类映射 (硬编码，直接查表)
OP_CLASSIFICATION = {
    # ====== Softmax类 ======
    "softmax":     {"category": "softmax",    "complexity": "medium",  "template": "01_SOFTMAX_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceMax", "Adds", "Exp", "ReduceSum", "Muls"]},
    "log_softmax": {"category": "softmax",    "complexity": "medium",  "template": "01_SOFTMAX_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceMax", "Adds", "Exp", "ReduceSum", "Muls", "Log"]},
    "softmin":     {"category": "softmax",    "complexity": "medium",  "template": "01_SOFTMAX_TEMPLATE.md",
                    "apis": ["DataCopy", "Neg", "ReduceMax", "Adds", "Exp", "ReduceSum", "Muls"]},

    # ====== 一元算子 ======
    "exp":         {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Exp"]},
    "log":         {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Log"]},
    "sqrt":        {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Sqrt"]},
    "rsqrt":       {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Rsqrt"]},
    "abs":         {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Abs"]},
    "neg":         {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Neg"]},
    "sin":         {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Sin"]},
    "cos":         {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Cos"]},
    "tanh":        {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Tanh"]},
    "erf":         {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Erf"]},
    "reciprocal":  {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Reciprocal"]},
    "ceil":        {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Ceil"]},
    "floor":       {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Floor"]},
    "round":       {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Round"]},
    "sign":        {"category": "unary",      "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Sign"]},

    # ====== 激活函数 ======
    "relu":        {"category": "activation", "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Relu"]},
    "gelu":        {"category": "activation", "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Gelu"],
                    "also_see": "07_GELU_VARIANTS.md"},
    "silu":        {"category": "activation", "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Swish"]},
    "swish":       {"category": "activation", "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Swish"]},
    "sigmoid":     {"category": "activation", "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Sigmoid"]},
    "leaky_relu":  {"category": "activation", "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "LeakyRelu"]},
    "softplus":    {"category": "activation", "complexity": "simple",  "template": "02_UNARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Softplus"]},

    # ====== GELU变体 ======
    "gelu_tanh":   {"category": "activation", "complexity": "medium",  "template": "07_GELU_VARIANTS.md",
                    "apis": ["DataCopy", "Mul", "Muls", "Add", "Adds", "Tanh"]},
    "quick_gelu":  {"category": "activation", "complexity": "simple",  "template": "07_GELU_VARIANTS.md",
                    "apis": ["DataCopy", "Muls", "Sigmoid", "Mul"]},
    "geglu":       {"category": "gated",      "complexity": "medium",  "template": "07_GELU_VARIANTS.md",
                    "apis": ["DataCopy", "Gelu", "Mul"]},
    "swiglu":      {"category": "gated",      "complexity": "medium",  "template": "07_GELU_VARIANTS.md",
                    "apis": ["DataCopy", "Swish", "Mul"]},

    # ====== 二元算子 ======
    "add":         {"category": "binary",     "complexity": "simple",  "template": "03_BINARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Add"]},
    "sub":         {"category": "binary",     "complexity": "simple",  "template": "03_BINARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Sub"]},
    "mul":         {"category": "binary",     "complexity": "simple",  "template": "03_BINARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Mul"]},
    "div":         {"category": "binary",     "complexity": "simple",  "template": "03_BINARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Div"]},
    "pow":         {"category": "binary",     "complexity": "simple",  "template": "03_BINARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Pow"]},
    "maximum":     {"category": "binary",     "complexity": "simple",  "template": "03_BINARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Max"]},
    "minimum":     {"category": "binary",     "complexity": "simple",  "template": "03_BINARY_TEMPLATE.md",
                    "apis": ["DataCopy", "Min"]},

    # ====== Norm类 ======
    "layernorm":   {"category": "norm",       "complexity": "medium",  "template": "04_NORM_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceSum", "Muls", "Adds", "Mul", "Add", "Sqrt"]},
    "layer_norm":  {"category": "norm",       "complexity": "medium",  "template": "04_NORM_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceSum", "Muls", "Adds", "Mul", "Add", "Sqrt"]},
    "rmsnorm":     {"category": "norm",       "complexity": "medium",  "template": "04_NORM_TEMPLATE.md",
                    "apis": ["DataCopy", "Mul", "ReduceSum", "Muls", "Rsqrt"]},
    "rms_norm":    {"category": "norm",       "complexity": "medium",  "template": "04_NORM_TEMPLATE.md",
                    "apis": ["DataCopy", "Mul", "ReduceSum", "Muls", "Rsqrt"]},
    "batchnorm":   {"category": "norm",       "complexity": "medium",  "template": "04_NORM_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceSum", "Muls", "Sqrt", "Mul", "Add"]},
    "batch_norm":  {"category": "norm",       "complexity": "medium",  "template": "04_NORM_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceSum", "Muls", "Sqrt", "Mul", "Add"]},
    "groupnorm":   {"category": "norm",       "complexity": "medium",  "template": "04_NORM_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceSum", "Muls", "Sqrt", "Mul", "Add"]},
    "instancenorm": {"category": "norm",      "complexity": "medium",  "template": "04_NORM_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceSum", "Muls", "Sqrt", "Mul", "Add"]},

    # ====== Reduce类 ======
    "reduce_sum":  {"category": "reduce",     "complexity": "medium",  "template": "05_REDUCE_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceSum"]},
    "reduce_mean": {"category": "reduce",     "complexity": "medium",  "template": "05_REDUCE_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceSum", "Muls"]},
    "reduce_max":  {"category": "reduce",     "complexity": "medium",  "template": "05_REDUCE_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceMax"]},
    "reduce_min":  {"category": "reduce",     "complexity": "medium",  "template": "05_REDUCE_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceMin"]},
    "sum":         {"category": "reduce",     "complexity": "medium",  "template": "05_REDUCE_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceSum"]},
    "mean":        {"category": "reduce",     "complexity": "medium",  "template": "05_REDUCE_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceSum", "Muls"]},
    "argmax":      {"category": "reduce",     "complexity": "medium",  "template": "05_REDUCE_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceMax"]},
    "argmin":      {"category": "reduce",     "complexity": "medium",  "template": "05_REDUCE_TEMPLATE.md",
                    "apis": ["DataCopy", "ReduceMin"]},

    # ====== MatMul类 ======
    "matmul":      {"category": "matmul",     "complexity": "complex", "template": "06_MATMUL_TEMPLATE.md",
                    "apis": ["DataCopy", "Mmad"]},
    "gemm":        {"category": "matmul",     "complexity": "complex", "template": "06_MATMUL_TEMPLATE.md",
                    "apis": ["DataCopy", "Mmad"]},
    "linear":      {"category": "matmul",     "complexity": "complex", "template": "06_MATMUL_TEMPLATE.md",
                    "apis": ["DataCopy", "Mmad"]},
    "bmm":         {"category": "matmul",     "complexity": "complex", "template": "06_MATMUL_TEMPLATE.md",
                    "apis": ["DataCopy", "Mmad"]},

    # ====== Broadcast类 ======
    "add_scalar":  {"category": "broadcast",  "complexity": "simple",  "template": "08_BROADCAST_TEMPLATE.md",
                    "apis": ["DataCopy", "Adds"]},
    "mul_scalar":  {"category": "broadcast",  "complexity": "simple",  "template": "08_BROADCAST_TEMPLATE.md",
                    "apis": ["DataCopy", "Muls"]},
    "bias_add":    {"category": "broadcast",  "complexity": "simple",  "template": "08_BROADCAST_TEMPLATE.md",
                    "apis": ["DataCopy", "Add"]},
    "scale":       {"category": "broadcast",  "complexity": "simple",  "template": "08_BROADCAST_TEMPLATE.md",
                    "apis": ["DataCopy", "Mul"]},

    # ====== Attention类 ======
    "attention":             {"category": "attention", "complexity": "complex", "template": "09_ATTENTION_TEMPLATE.md",
                              "apis": ["DataCopy", "Mmad", "Muls", "ReduceMax", "Exp", "ReduceSum"]},
    "self_attention":        {"category": "attention", "complexity": "complex", "template": "09_ATTENTION_TEMPLATE.md",
                              "apis": ["DataCopy", "Mmad", "Muls", "ReduceMax", "Exp", "ReduceSum"]},
    "scaled_dot_attention":  {"category": "attention", "complexity": "complex", "template": "09_ATTENTION_TEMPLATE.md",
                              "apis": ["DataCopy", "Mmad", "Muls", "ReduceMax", "Exp", "ReduceSum"]},
    "flash_attention":       {"category": "attention", "complexity": "complex", "template": "09_ATTENTION_TEMPLATE.md",
                              "apis": ["DataCopy", "Mmad", "Muls", "ReduceMax", "Exp", "ReduceSum"]},

    # ====== 融合算子 ======
    "add_relu":         {"category": "fused",  "complexity": "simple",  "template": "10_FUSED_TEMPLATE.md",
                         "apis": ["DataCopy", "Add", "Relu"]},
    "add_gelu":         {"category": "fused",  "complexity": "simple",  "template": "10_FUSED_TEMPLATE.md",
                         "apis": ["DataCopy", "Add", "Gelu"]},
    "bias_gelu":        {"category": "fused",  "complexity": "simple",  "template": "10_FUSED_TEMPLATE.md",
                         "apis": ["DataCopy", "Add", "Gelu"]},
    "mul_add":          {"category": "fused",  "complexity": "simple",  "template": "10_FUSED_TEMPLATE.md",
                         "apis": ["DataCopy", "Mul", "Add"]},
    "residual_add":     {"category": "fused",  "complexity": "simple",  "template": "10_FUSED_TEMPLATE.md",
                         "apis": ["DataCopy", "Add"]},
    "add_layernorm":    {"category": "fused",  "complexity": "medium",  "template": "10_FUSED_TEMPLATE.md",
                         "apis": ["DataCopy", "Add", "ReduceSum", "Muls", "Adds", "Mul", "Sqrt"]},
    "residual_layernorm": {"category": "fused", "complexity": "medium", "template": "10_FUSED_TEMPLATE.md",
                         "apis": ["DataCopy", "Add", "ReduceSum", "Muls", "Adds", "Mul", "Sqrt"]},
}


def lookup(op_name: str):
    """查询算子信息"""
    op_lower = op_name.lower().replace("-", "_")

    # 直接匹配
    if op_lower in OP_CLASSIFICATION:
        return OP_CLASSIFICATION[op_lower]

    # 部分匹配
    for key, info in OP_CLASSIFICATION.items():
        if key in op_lower or op_lower in key:
            return info

    # 未找到
    return {
        "category": "unknown",
        "complexity": "unknown",
        "template": "02_UNARY_TEMPLATE.md",
        "apis": ["DataCopy"],
        "warning": "未找到精确匹配，默认使用一元模板"
    }


def list_all():
    """列出所有支持的算子"""
    categories = {}
    for op, info in sorted(OP_CLASSIFICATION.items()):
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(op)

    print(f"\n支持的算子 (共 {len(OP_CLASSIFICATION)} 个):")
    print("=" * 60)
    for cat, ops in sorted(categories.items()):
        print(f"\n  [{cat}]")
        line = "    "
        for op in ops:
            if len(line) + len(op) > 58:
                print(line)
                line = "    "
            line += op + ", "
        if line.strip():
            print(line.rstrip(", "))


def main():
    if len(sys.argv) < 2:
        print("用法: python op_lookup.py <算子名>")
        print("      python op_lookup.py list")
        print("示例: python op_lookup.py softmax")
        return

    cmd = sys.argv[1]

    if cmd == "list":
        list_all()
        return

    info = lookup(cmd)

    print(f"\n{'='*50}")
    print(f"算子: {cmd}")
    print(f"{'='*50}")
    print(f"类别:     {info['category']}")
    print(f"复杂度:   {info['complexity']}")
    print(f"参考模板: templates/{info['template']}")
    print(f"所需API:  {', '.join(info['apis'])}")

    if info.get("also_see"):
        print(f"另见:     templates/{info['also_see']}")

    if info.get("warning"):
        print(f"\n⚠️  {info['warning']}")

    print(f"\n下一步: cat templates/{info['template']}")


if __name__ == "__main__":
    main()

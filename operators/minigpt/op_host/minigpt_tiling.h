/**
 * MiniGPT Tiling 数据结构定义
 */
#ifndef MINIGPT_TILING_H
#define MINIGPT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// GELU Tiling
BEGIN_TILING_DATA_DEF(GeluTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GeluCustom, GeluTilingData)

// LayerNorm Tiling
BEGIN_TILING_DATA_DEF(LayerNormTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);
    TILING_DATA_FIELD_DEF(uint32_t, rowSize);
    TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(LayerNormCustom, LayerNormTilingData)

// Softmax Tiling
BEGIN_TILING_DATA_DEF(SoftmaxTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);
    TILING_DATA_FIELD_DEF(uint32_t, rowSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SoftmaxCustom, SoftmaxTilingData)

// MatMul Tiling
BEGIN_TILING_DATA_DEF(MatmulTilingData)
    TILING_DATA_FIELD_DEF(int32_t, M);
    TILING_DATA_FIELD_DEF(int32_t, N);
    TILING_DATA_FIELD_DEF(int32_t, K);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulCustom, MatmulTilingData)

// Add Tiling
BEGIN_TILING_DATA_DEF(AddTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AddCustom, AddTilingData)

// Embedding Tiling
BEGIN_TILING_DATA_DEF(EmbeddingTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, seqLen);
    TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
    TILING_DATA_FIELD_DEF(uint32_t, hiddenDim);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EmbeddingCustom, EmbeddingTilingData)

}  // namespace optiling

#endif  // MINIGPT_TILING_H

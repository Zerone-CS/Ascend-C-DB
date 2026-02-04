/*
 * MatMul Custom Operator - Tiling Data Definition
 * 
 * 定义与 TCubeTiling 结构相同的自定义 tiling 结构
 */
#ifndef MATMUL_CUSTOM_TILING_H
#define MATMUL_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// 简单的 tiling 结构，仅包含必要字段
BEGIN_TILING_DATA_DEF(MatmulCustomTilingData)
    TILING_DATA_FIELD_DEF(int32_t, M);
    TILING_DATA_FIELD_DEF(int32_t, N);
    TILING_DATA_FIELD_DEF(int32_t, K);
    TILING_DATA_FIELD_DEF(int32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulCustom, MatmulCustomTilingData)

}  // namespace optiling

#endif  // MATMUL_CUSTOM_TILING_H

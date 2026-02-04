/**
 * Tiling 结构体定义
 * Host 端计算，Kernel 端使用
 */
#ifndef RELU_CUSTOM_TILING_H
#define RELU_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ReluCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);  // 总元素数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReluCustom, ReluCustomTilingData)

}
#endif

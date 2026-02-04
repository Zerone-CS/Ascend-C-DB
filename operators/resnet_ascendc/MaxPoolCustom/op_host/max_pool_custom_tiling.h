#ifndef MAX_POOL_CUSTOM_TILING_H
#define MAX_POOL_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MaxPoolCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, pad);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MaxPoolCustom, MaxPoolCustomTilingData)
}
#endif

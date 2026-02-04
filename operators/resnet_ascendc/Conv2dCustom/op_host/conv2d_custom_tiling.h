#ifndef CONV2D_CUSTOM_TILING_H
#define CONV2D_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, inC);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, outC);
    TILING_DATA_FIELD_DEF(uint32_t, kH);
    TILING_DATA_FIELD_DEF(uint32_t, kW);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, pad);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Conv2dCustom, Conv2dCustomTilingData)
}
#endif

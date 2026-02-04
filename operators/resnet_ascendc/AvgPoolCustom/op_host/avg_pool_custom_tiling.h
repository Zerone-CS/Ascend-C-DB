#ifndef AVG_POOL_CUSTOM_TILING_H
#define AVG_POOL_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(AvgPoolCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AvgPoolCustom, AvgPoolCustomTilingData)
}
#endif

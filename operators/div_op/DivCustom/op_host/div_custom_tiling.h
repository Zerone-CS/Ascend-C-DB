#ifndef DIV_CUSTOM_TILING_H
#define DIV_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DivCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DivCustom, DivCustomTilingData)
}
#endif

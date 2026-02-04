#ifndef FC_CUSTOM_TILING_H
#define FC_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(FcCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FcCustom, FcCustomTilingData)
}
#endif

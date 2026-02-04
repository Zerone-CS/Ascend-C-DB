#ifndef SIGMOID_CUSTOM_TILING_H
#define SIGMOID_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SigmoidCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SigmoidCustom, SigmoidCustomTilingData)
}
#endif

#ifndef RESNET_OPS_TILING_H
#define RESNET_OPS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {

// ReLU
BEGIN_TILING_DATA_DEF(ReluCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ReluCustom, ReluCustomTilingData)

// Add
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)

// Conv2D
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

// BatchNorm
BEGIN_TILING_DATA_DEF(BnCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BnCustom, BnCustomTilingData)

// MaxPool
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

// AvgPool (Global)
BEGIN_TILING_DATA_DEF(AvgPoolCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, HW);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AvgPoolCustom, AvgPoolCustomTilingData)

// FC
BEGIN_TILING_DATA_DEF(FcCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FcCustom, FcCustomTilingData)

}
#endif

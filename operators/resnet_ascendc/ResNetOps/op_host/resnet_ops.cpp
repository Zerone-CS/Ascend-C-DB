/**
 * ResNet 所有自定义算子 - Host 端注册
 */
#include "resnet_ops_tiling.h"
#include "register/op_def_registry.h"

// ========== Tiling Functions ==========
namespace optiling {

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context) {
    ReluCustomTilingData tiling;
    const gert::StorageShape* shape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++)
        totalLength *= shape->GetStorageShape().GetDim(i);
    tiling.set_totalLength(totalLength);
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AddTilingFunc(gert::TilingContext* context) {
    AddCustomTilingData tiling;
    const gert::StorageShape* shape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++)
        totalLength *= shape->GetStorageShape().GetDim(i);
    tiling.set_totalLength(totalLength);
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Conv2dTilingFunc(gert::TilingContext* context) {
    Conv2dCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* wShape = context->GetInputShape(1);
    tiling.set_N(xShape->GetStorageShape().GetDim(0));
    tiling.set_inC(xShape->GetStorageShape().GetDim(1));
    tiling.set_H(xShape->GetStorageShape().GetDim(2));
    tiling.set_W(xShape->GetStorageShape().GetDim(3));
    tiling.set_outC(wShape->GetStorageShape().GetDim(0));
    tiling.set_kH(wShape->GetStorageShape().GetDim(2));
    tiling.set_kW(wShape->GetStorageShape().GetDim(3));
    const int64_t* sAttr = context->GetAttrs()->GetInt(0);
    const int64_t* pAttr = context->GetAttrs()->GetInt(1);
    tiling.set_stride(sAttr ? *sAttr : 1);
    tiling.set_pad(pAttr ? *pAttr : 0);
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BnTilingFunc(gert::TilingContext* context) {
    BnCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    tiling.set_N(xShape->GetStorageShape().GetDim(0));
    tiling.set_C(xShape->GetStorageShape().GetDim(1));
    uint32_t HW = 1;
    for (int i = 2; i < xShape->GetStorageShape().GetDimNum(); i++)
        HW *= xShape->GetStorageShape().GetDim(i);
    tiling.set_HW(HW);
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MaxPoolTilingFunc(gert::TilingContext* context) {
    MaxPoolCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    tiling.set_N(xShape->GetStorageShape().GetDim(0));
    tiling.set_C(xShape->GetStorageShape().GetDim(1));
    tiling.set_H(xShape->GetStorageShape().GetDim(2));
    tiling.set_W(xShape->GetStorageShape().GetDim(3));
    const int64_t* kAttr = context->GetAttrs()->GetInt(0);
    const int64_t* sAttr = context->GetAttrs()->GetInt(1);
    const int64_t* pAttr = context->GetAttrs()->GetInt(2);
    tiling.set_K(kAttr ? *kAttr : 3);
    tiling.set_stride(sAttr ? *sAttr : 2);
    tiling.set_pad(pAttr ? *pAttr : 0);
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AvgPoolTilingFunc(gert::TilingContext* context) {
    AvgPoolCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    tiling.set_N(xShape->GetStorageShape().GetDim(0));
    tiling.set_C(xShape->GetStorageShape().GetDim(1));
    uint32_t HW = 1;
    for (int i = 2; i < xShape->GetStorageShape().GetDimNum(); i++)
        HW *= xShape->GetStorageShape().GetDim(i);
    tiling.set_HW(HW);
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FcTilingFunc(gert::TilingContext* context) {
    FcCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* wShape = context->GetInputShape(1);
    tiling.set_M(xShape->GetStorageShape().GetDim(0));
    tiling.set_K(xShape->GetStorageShape().GetDim(1));
    tiling.set_N(wShape->GetStorageShape().GetDim(1));
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

// ========== Shape/DataType Inference ==========
namespace ge {

static graphStatus InferShapeSame(gert::InferShapeContext* context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeSame(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeConv2d(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* wShape = context->GetInputShape(1);
    int64_t N = xShape->GetDim(0), H = xShape->GetDim(2), W = xShape->GetDim(3);
    int64_t outC = wShape->GetDim(0), kH = wShape->GetDim(2), kW = wShape->GetDim(3);
    const int64_t* sAttr = context->GetAttrs()->GetInt(0);
    const int64_t* pAttr = context->GetAttrs()->GetInt(1);
    int64_t stride = sAttr ? *sAttr : 1;
    int64_t pad = pAttr ? *pAttr : 0;
    gert::Shape* out = context->GetOutputShape(0);
    out->SetDimNum(4);
    out->SetDim(0, N); out->SetDim(1, outC);
    out->SetDim(2, (H + 2*pad - kH) / stride + 1);
    out->SetDim(3, (W + 2*pad - kW) / stride + 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeMaxPool(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    int64_t N = xShape->GetDim(0), C = xShape->GetDim(1);
    int64_t H = xShape->GetDim(2), W = xShape->GetDim(3);
    const int64_t* kAttr = context->GetAttrs()->GetInt(0);
    const int64_t* sAttr = context->GetAttrs()->GetInt(1);
    const int64_t* pAttr = context->GetAttrs()->GetInt(2);
    int64_t K = kAttr ? *kAttr : 3, S = sAttr ? *sAttr : 2, P = pAttr ? *pAttr : 0;
    gert::Shape* out = context->GetOutputShape(0);
    out->SetDimNum(4); out->SetDim(0, N); out->SetDim(1, C);
    out->SetDim(2, (H + 2*P - K) / S + 1);
    out->SetDim(3, (W + 2*P - K) / S + 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeAvgPool(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* out = context->GetOutputShape(0);
    out->SetDimNum(2); out->SetDim(0, xShape->GetDim(0)); out->SetDim(1, xShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeFc(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* wShape = context->GetInputShape(1);
    gert::Shape* out = context->GetOutputShape(0);
    out->SetDimNum(2); out->SetDim(0, xShape->GetDim(0)); out->SetDim(1, wShape->GetDim(1));
    return GRAPH_SUCCESS;
}

} // namespace ge

// ========== Operator Registration ==========
namespace ops {

class ReluCustom : public OpDef {
public:
    explicit ReluCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShapeSame).SetInferDataType(ge::InferDataTypeSame);
        this->AICore().SetTiling(optiling::ReluTilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ReluCustom);

class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name) {
        this->Input("x1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("x2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShapeSame).SetInferDataType(ge::InferDataTypeSame);
        this->AICore().SetTiling(optiling::AddTilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(AddCustom);

class Conv2dCustom : public OpDef {
public:
    explicit Conv2dCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("stride").AttrType(OPTIONAL).Int(1);
        this->Attr("pad").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(ge::InferShapeConv2d).SetInferDataType(ge::InferDataTypeSame);
        this->AICore().SetTiling(optiling::Conv2dTilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Conv2dCustom);

class BnCustom : public OpDef {
public:
    explicit BnCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("mean").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("var").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShapeSame).SetInferDataType(ge::InferDataTypeSame);
        this->AICore().SetTiling(optiling::BnTilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(BnCustom);

class MaxPoolCustom : public OpDef {
public:
    explicit MaxPoolCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("ksize").AttrType(OPTIONAL).Int(3);
        this->Attr("stride").AttrType(OPTIONAL).Int(2);
        this->Attr("pad").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(ge::InferShapeMaxPool).SetInferDataType(ge::InferDataTypeSame);
        this->AICore().SetTiling(optiling::MaxPoolTilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(MaxPoolCustom);

class AvgPoolCustom : public OpDef {
public:
    explicit AvgPoolCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShapeAvgPool).SetInferDataType(ge::InferDataTypeSame);
        this->AICore().SetTiling(optiling::AvgPoolTilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(AvgPoolCustom);

class FcCustom : public OpDef {
public:
    explicit FcCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShapeFc).SetInferDataType(ge::InferDataTypeSame);
        this->AICore().SetTiling(optiling::FcTilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(FcCustom);

} // namespace ops

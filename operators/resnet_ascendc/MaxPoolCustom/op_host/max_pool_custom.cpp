#include "max_pool_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
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
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    int64_t N = xShape->GetDim(0), C = xShape->GetDim(1);
    int64_t H = xShape->GetDim(2), W = xShape->GetDim(3);
    const int64_t* kAttr = context->GetAttrs()->GetInt(0);
    const int64_t* sAttr = context->GetAttrs()->GetInt(1);
    const int64_t* pAttr = context->GetAttrs()->GetInt(2);
    int64_t K = kAttr ? *kAttr : 3;
    int64_t S = sAttr ? *sAttr : 2;
    int64_t P = pAttr ? *pAttr : 0;
    gert::Shape* out = context->GetOutputShape(0);
    out->SetDimNum(4);
    out->SetDim(0, N); out->SetDim(1, C);
    out->SetDim(2, (H + 2*P - K) / S + 1);
    out->SetDim(3, (W + 2*P - K) / S + 1);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class MaxPoolCustom : public OpDef {
public:
    explicit MaxPoolCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("ksize").AttrType(OPTIONAL).Int(3);
        this->Attr("stride").AttrType(OPTIONAL).Int(2);
        this->Attr("pad").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(MaxPoolCustom);
}

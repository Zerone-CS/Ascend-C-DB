#include "avg_pool_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    AvgPoolCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t N = xShape->GetStorageShape().GetDim(0);
    uint32_t C = xShape->GetStorageShape().GetDim(1);
    uint32_t HW = 1;
    for (int i = 2; i < xShape->GetStorageShape().GetDimNum(); i++)
        HW *= xShape->GetStorageShape().GetDim(i);
    tiling.set_N(N); tiling.set_C(C); tiling.set_HW(HW);
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
    gert::Shape* out = context->GetOutputShape(0);
    out->SetDimNum(2);
    out->SetDim(0, xShape->GetDim(0));
    out->SetDim(1, xShape->GetDim(1));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AvgPoolCustom : public OpDef {
public:
    explicit AvgPoolCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(AvgPoolCustom);
}

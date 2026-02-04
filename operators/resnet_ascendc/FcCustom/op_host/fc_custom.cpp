#include "fc_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
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
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* wShape = context->GetInputShape(1);
    gert::Shape* out = context->GetOutputShape(0);
    out->SetDimNum(2);
    out->SetDim(0, xShape->GetDim(0));
    out->SetDim(1, wShape->GetDim(1));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class FcCustom : public OpDef {
public:
    explicit FcCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(FcCustom);
}

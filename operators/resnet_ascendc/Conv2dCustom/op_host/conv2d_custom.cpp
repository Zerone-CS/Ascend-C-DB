#include "conv2d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    Conv2dCustomTilingData tiling;
    
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* wShape = context->GetInputShape(1);
    
    uint32_t N = xShape->GetStorageShape().GetDim(0);
    uint32_t inC = xShape->GetStorageShape().GetDim(1);
    uint32_t H = xShape->GetStorageShape().GetDim(2);
    uint32_t W = xShape->GetStorageShape().GetDim(3);
    uint32_t outC = wShape->GetStorageShape().GetDim(0);
    uint32_t kH = wShape->GetStorageShape().GetDim(2);
    uint32_t kW = wShape->GetStorageShape().GetDim(3);
    
    uint32_t stride = 1;
    uint32_t pad = 0;
    const int64_t* strideAttr = context->GetAttrs()->GetInt(0);
    const int64_t* padAttr = context->GetAttrs()->GetInt(1);
    if (strideAttr) stride = *strideAttr;
    if (padAttr) pad = *padAttr;
    
    tiling.set_N(N);
    tiling.set_inC(inC);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_outC(outC);
    tiling.set_kH(kH);
    tiling.set_kW(kW);
    tiling.set_stride(stride);
    tiling.set_pad(pad);
    
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
    
    int64_t N = xShape->GetDim(0);
    int64_t H = xShape->GetDim(2);
    int64_t W = xShape->GetDim(3);
    int64_t outC = wShape->GetDim(0);
    int64_t kH = wShape->GetDim(2);
    int64_t kW = wShape->GetDim(3);
    
    int64_t stride = 1;
    int64_t pad = 0;
    const int64_t* strideAttr = context->GetAttrs()->GetInt(0);
    const int64_t* padAttr = context->GetAttrs()->GetInt(1);
    if (strideAttr) stride = *strideAttr;
    if (padAttr) pad = *padAttr;
    
    int64_t outH = (H + 2 * pad - kH) / stride + 1;
    int64_t outW = (W + 2 * pad - kW) / stride + 1;
    
    gert::Shape* outShape = context->GetOutputShape(0);
    outShape->SetDimNum(4);
    outShape->SetDim(0, N);
    outShape->SetDim(1, outC);
    outShape->SetDim(2, outH);
    outShape->SetDim(3, outW);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv2dCustom : public OpDef {
public:
    explicit Conv2dCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("stride").AttrType(OPTIONAL).Int(1);
        this->Attr("pad").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Conv2dCustom);
}

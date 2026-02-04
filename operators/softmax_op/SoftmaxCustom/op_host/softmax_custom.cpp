#include "softmax_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SoftmaxCustomTilingData tiling;
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    
    // Get shape dimensions
    int32_t dimNum = x_shape->GetStorageShape().GetDimNum();
    uint32_t totalRows = 1;
    uint32_t rowSize = 1;
    
    if (dimNum >= 2) {
        // Last dimension is row size (softmax over last axis)
        rowSize = x_shape->GetStorageShape().GetDim(dimNum - 1);
        // All other dimensions multiplied are total rows
        for (int i = 0; i < dimNum - 1; i++) {
            totalRows *= x_shape->GetStorageShape().GetDim(i);
        }
    } else if (dimNum == 1) {
        totalRows = 1;
        rowSize = x_shape->GetStorageShape().GetDim(0);
    }
    
    tiling.set_totalRows(totalRows);
    tiling.set_rowSize(rowSize);
    
    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class SoftmaxCustom : public OpDef {
public:
    explicit SoftmaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SoftmaxCustom);
}
